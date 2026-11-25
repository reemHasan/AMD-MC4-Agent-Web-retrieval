"""
Resident server. Started once by the container CMD:  python3 /app/server.py
Loads the model + browser once, then answers one question at a time over a unix socket.
Request : {"url": "...", "query": "..."}\n
Response: {"answer": "...", "citations": [...], "confidence": 0.9}\n
"""
import json, os, re, socketserver, time
from urllib.parse import urljoin, urlparse

import httpx
import torch
from bs4 import BeautifulSoup
from transformers import AutoModelForImageTextToText, AutoProcessor

SOCK = "/tmp/agent.sock"
MODEL_DIR = os.environ.get("MC3_MODEL_DIR", "/models/vlm")
DEBUG = os.environ.get("AGENT_DEBUG") == "1"
MAX_STEPS = 6
DEADLINE = 24          # seconds per question (hard limit is 30)
CTX_CHARS = 5000       # page text sent to the model per step

EMPTY = {"answer": "", "citations": []}

SYSTEM = (
    "You extract answers from web pages. Answer ONLY from the page text shown; never use "
    "outside knowledge. 'value' is the bare value only (e.g. '$49.99', 'Q3 FY27'), never a "
    "sentence. 'evidence' is an EXACT quote copied from the page text that contains the value. "
    "If the page does not contain the answer, set found=false and pick the most promising "
    "'next_url' from the link list (empty string if none).\n"
    'Reply with ONLY a JSON object: {"found": bool, "value": str, "evidence": str, "next_url": str}'
)

# ---------------------------------------------------------------- model
print("loading model from", MODEL_DIR, "| torch", torch.__version__, "| gpu", torch.cuda.is_available(), flush=True)
processor = AutoProcessor.from_pretrained(MODEL_DIR)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cuda"
).eval()


def generate(system, user, max_new_tokens=300):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": [{"type": "text", "text": user}]},
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def llm_json(system, user):
    """Ask for JSON, parse it, retry once on failure."""
    for _ in range(2):
        raw = generate(system, user)
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            try:
                d = json.loads(m.group(0))
                return {
                    "found": bool(d.get("found")),
                    "value": str(d.get("value", "")),
                    "evidence": str(d.get("evidence", "")),
                    "next_url": str(d.get("next_url", "")),
                }
            except Exception:
                pass
    return None


# ---------------------------------------------------------------- browser (optional)
BROWSER = None
try:
    from playwright.sync_api import sync_playwright
    _pw = sync_playwright().start()
    BROWSER = _pw.chromium.launch(headless=True)
    print("chromium ready", flush=True)
except Exception as e:
    print("chromium unavailable, HTTP-only mode:", e, flush=True)


def fetch_html(url):
    try:
        r = httpx.get(url, timeout=8, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"})
        html = r.text
        if len(BeautifulSoup(html, "html.parser").get_text(" ", strip=True)) > 400 or BROWSER is None:
            return html
    except Exception:
        if BROWSER is None:
            return ""
    try:  # JS-heavy page: render it
        page = BROWSER.new_page()
        try:
            page.goto(url, timeout=12000, wait_until="networkidle")
            return page.content()
        finally:
            page.close()
    except Exception:
        return ""


def parse(url, html):
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "noscript", "svg", "footer"]):
        t.decompose()
    for a in soup.find_all("a", title=True):  # visible text is often truncated; title has the full text
        a.string = a["title"]
    for tr in soup.find_all("tr"):  # keep table rows readable
        tr.append(" | ".join(c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])) + "\n")
    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
    links, seen = [], set()
    for a in soup.find_all("a", href=True):
        u = urljoin(url, a["href"]).split("#")[0]
        if urlparse(u).netloc == urlparse(url).netloc and u not in seen and u.startswith("http"):
            seen.add(u)
            links.append((a.get_text(" ", strip=True)[:60], u))
    return text, links


# ---------------------------------------------------------------- helpers
def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def tokens(s):
    return set(re.findall(r"\w{3,}", s.lower()))


def select_context(text, query, limit=CTX_CHARS, size=700):
    """Keep the chunks most related to the query (lexical overlap), in original order."""
    if len(text) <= limit:
        return text
    chunks = [text[i:i + size] for i in range(0, len(text), size)]
    q = tokens(query)
    ranked = sorted(range(len(chunks)), key=lambda i: -len(q & tokens(chunks[i])))
    keep, total = [], 0
    for i in ranked:
        if total + len(chunks[i]) > limit:
            break
        keep.append(i)
        total += len(chunks[i])
    return " ... ".join(chunks[i] for i in sorted(keep))


def rank_links(links, query, k=40):
    q = tokens(query)
    return sorted(links, key=lambda l: -len(q & tokens(l[0] + " " + l[1])))[:k]


# ---------------------------------------------------------------- agent
def answer(url, query):
    start, visited = time.time(), set()
    for _ in range(MAX_STEPS):
        if time.time() - start > DEADLINE or not url or url in visited:
            break
        visited.add(url)
        text, links = parse(url, fetch_html(url))
        if not text:
            if DEBUG:
                print(f"[step] empty text for {url}", flush=True)
            break
        link_list = "\n".join(f"- {t} -> {u}" for t, u in rank_links(links, query))
        user = f"Question: {query}\n\nPAGE URL: {url}\nPAGE TEXT:\n{select_context(text, query)}\n\nLINKS:\n{link_list}"
        out = llm_json(SYSTEM, user)
        if DEBUG:
            print(f"[step] url={url} text_len={len(text)} out={out}", flush=True)
        if out is None:
            break
        # Verification gate: evidence must literally appear on the page and contain the value
        if out["found"] and out["value"].strip() and norm(out["evidence"]) in norm(text) \
                and norm(out["value"]) in norm(out["evidence"]):
            # TODO multi-hop: cite only pages that contributed a needed fact
            return {"answer": out["value"].strip(), "citations": [url], "confidence": 0.9}
        if out["found"]:
            break  # claimed an answer but it failed verification -> abstain
        url = urljoin(url, out["next_url"]) if out["next_url"] else ""
    return dict(EMPTY)


# ---------------------------------------------------------------- socket server
class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            req = json.loads(self.rfile.readline())
            res = answer(req["url"], req["query"])
        except Exception as e:
            print("error:", e, flush=True)
            res = dict(EMPTY)
        self.wfile.write((json.dumps(res) + "\n").encode())


if __name__ == "__main__":
    if os.path.exists(SOCK):
        os.remove(SOCK)
    with socketserver.UnixStreamServer(SOCK, Handler) as s:  # single-threaded on purpose
        print("ready", flush=True)
        s.serve_forever()