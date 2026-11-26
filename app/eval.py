"""
Evaluate the agent on a test set. The resident server must already be running.

    python3 eval.py [tests.json]

Test case fields:
  id        short name
  query     the question
  url       optional start URL (omit to test open mode)
  expected  the answer value, or "" when the answer is NOT available (must abstain)
  cite      optional substring (or list of substrings) that must appear in the citations
"""
import json, re, statistics, sys, time, unicodedata
from app import ask


def norm(s):
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip(" .,;:")


def check(case, res):
    got, cites = res.get("answer", ""), res.get("citations", [])
    exp = case["expected"]
    if exp == "":
        ok = got == "" and not cites
        return ok, True, "abstained" if ok else f"should abstain, got {got!r}"
    if norm(got) != norm(exp):
        return False, False, f"answer {got!r} != {exp!r}"
    need = case.get("cite") or []
    need = [need] if isinstance(need, str) else need
    cite_ok = all(any(n in c for c in cites) for n in need)
    return True, cite_ok, "ok" if cite_ok else f"citations {cites} missing {need}"


def main(path):
    cases = json.load(open(path))
    rows = []
    for c in cases:
        t0 = time.time()
        res = ask(c.get("url", ""), c["query"])
        dt = time.time() - t0
        ans_ok, cite_ok, note = check(c, res)
        rows.append({"id": c["id"], "answerable": c["expected"] != "", "ans_ok": ans_ok,
                     "cite_ok": cite_ok, "sec": round(dt, 1), "note": note, "result": res})
        flag = "PASS" if ans_ok and cite_ok else "FAIL"
        print(f"{flag}  {c['id']:<10} {dt:5.1f}s  {note}")

    A = [r for r in rows if r["answerable"]]
    N = [r for r in rows if not r["answerable"]]
    full = [r for r in rows if r["ans_ok"] and r["cite_ok"]]
    times = sorted(r["sec"] for r in rows)
    print("\n==== summary ====")
    print(f"fully correct (answer + citations): {len(full)}/{len(rows)}")
    print(f"answerable questions, answer correct: {sum(r['ans_ok'] for r in A)}/{len(A)}")
    print(f"answerable questions, citations ok  : {sum(r['ans_ok'] and r['cite_ok'] for r in A)}/{len(A)}")
    print(f"unanswerable, correctly abstained   : {sum(r['ans_ok'] for r in N)}/{len(N)}")
    if times:
        p95 = times[min(len(times) - 1, int(len(times) * 0.95))]
        print(f"time per question: median {statistics.median(times):.1f}s, p95 {p95:.1f}s, max {times[-1]:.1f}s")
    json.dump(rows, open("eval_results.json", "w"), indent=2, ensure_ascii=False)
    print("details written to eval_results.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tests.json")