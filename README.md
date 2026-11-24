# Intelligent Web Information Retrieval Agent

An AI agent that takes a natural-language question, navigates a website, and returns **the specific value requested** together with the **exact page(s) it came from**. It runs fully offline inside a container on **AMD ROCm**, using a local vision-language model served in-process with `transformers`.

<!-->> **Status:** working prototype. Benchmarks below are placeholders to fill in after running the evaluation set.-->

## Key ideas

- **Answer only from the page.** The model never uses outside knowledge; if the site does not contain the answer, the agent returns an empty answer instead of guessing.
- **Verification gate.** An answer is accepted only if the model's evidence quote appears verbatim in the page text *and* contains the answer value. Anything else is discarded.
- **Resident server + thin client.** The model and browser load once at container start. Each question is a short-lived client process, so per-question latency is only inference plus page fetches.
- **Cheap first, browser second.** Pages are fetched with plain HTTP; Chromium is used only when a page looks JavaScript-rendered.

## Architecture

```
 question ──► app.py (thin client) ──unix socket──► server.py (resident)
                                                       │
                                  ┌────────────────────┴──────────────────────┐
                                  │ loop (max 6 steps / 24 s)                  │
                                  │ 1. fetch_html   HTTP, fallback Chromium    │
                                  │ 2. parse        clean text, tables, links  │
                                  │ 3. select       top chunks + ranked links  │
                                  │ 4. Qwen2.5-VL   JSON: found/value/evidence │
                                  │                 /next_url                  │
                                  │ 5. verify       evidence in page text?     │
                                  │    yes → answer + citation                 │
                                  │    no  → follow next_url or abstain        │
                                  └────────────────────────────────────────────┘
                                                       │
                              /app/output/<query-id>_output.json
```

### Files

| File | Role |
|------|------|
| `app.py` | Thin client. Sends the question to the server, waits if the model is still loading, **always** writes the output JSON. |
| `server.py` | Resident process: loads the model and browser once, runs the agent loop, serves requests over `/tmp/agent.sock`. |
| `requirements.txt` | Extra Python deps. Deliberately excludes `torch`/`transformers` (they come from the ROCm base image). |
| `Dockerfile` | Builds on the mandated ROCm PyTorch image, installs deps without replacing ROCm torch, bakes in Chromium and model weights. |

## Usage

### Command

```bash
python3 /app/app.py --url <START_URL> --query-id <ID> --query "<question>"
```

Writes `/app/output/<ID>_output.json`:

```json
{"answer": "£51.77", "citations": ["https://books.toscrape.com/..."], "confidence": 0.9}
```

If the answer is not on the site:

```json
{"answer": "", "citations": []}
```

### Build and run

1. Download the model into `models/vlm` (once, with network):

   ```bash
   python3 -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen2.5-VL-7B-Instruct', local_dir='models/vlm')"
   ```

2. Project layout expected by the Dockerfile:

   ```
   Dockerfile
   models/vlm/...          # model weights
   app/app.py
   app/server.py
   app/requirements.txt
   ```

3. Build and start (AMD GPU devices must be passed through):

   ```bash
   docker build -t web-agent .
   docker run -d --name agent \
     --device=/dev/kfd --device=/dev/dri --group-add video \
     --ipc=host --shm-size 16G web-agent
   ```

4. Wait for the server, then ask a question:

   ```bash
   docker exec agent cat /tmp/server.log          # look for "ready"
   docker exec agent python3 /app/app.py \
     --url https://books.toscrape.com \
     --query-id t1 --query "What is the price of A Light in the Attic?"
   docker exec agent cat /app/output/t1_output.json
   ```

## AMD ROCm usage

- Base image: `rocm/pytorch` (mandated), so all inference runs on the AMD GPU through ROCm.
- Packages are installed with `pip install --no-deps` where needed so pip never replaces ROCm PyTorch with a CUDA build. The build **fails** if `torch.__version__` does not contain `rocm`.
- At startup `server.py` logs the torch version and GPU availability to `/tmp/server.log`.

Evidence to include in your submission:

```bash
docker exec agent rocm-smi                      # GPU visible, memory in use
docker exec agent python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Design decisions

| Decision | Reason |
|----------|--------|
| Local 7B VLM in-process (no vLLM) | Matches the container model of the challenge; no second service to manage; the same weights can read screenshots later. |
| Evidence-quote verification | The main defence against hallucination when wrong answers score zero. |
| Abstain on failed verification | A wrong answer and a missing answer both score zero, but guessing risks contradicting the "no guessing" rule. |
| Single-threaded server | Playwright's sync API and one GPU model are not thread-safe. |
| Chromium optional | If it can't launch on the base OS, the agent still works for static sites in HTTP-only mode. |
| Lexical chunk and link ranking | Fast, dependency-free way to fit long pages and large link lists into a small context. |

## Known limitations

- **Single-page citations.** Multi-hop questions (answer needs facts from several pages) are not yet tracked with a provenance chain.
- **No interaction.** The agent follows links but does not type into search boxes, submit forms, or click buttons.
- **No PDF or image handling in the loop** yet (`pdfplumber` is installed; Qwen2.5-VL can read screenshots).
- **No answer normalization** beyond trimming whitespace.
- **Strict verification** (whitespace and case-insensitive exact match) may reject correct answers when the model paraphrases its quote.
- **Unverified performance.** Latency on a 7B model under plain `transformers` has not been measured yet.

## Roadmap

1. Multi-hop provenance: cite exactly the pages needed for the answer.
2. Search-box and form interaction via Playwright.
3. Embedding-based retrieval (e.g. BGE-M3) for chunks and links.
4. PDF extraction and screenshot reading with the VLM.
5. Answer normalization (units, dates, whitespace) tuned on the test set.
6. Global watchdog and per-step timing logs.

## Evaluation
To be filled ....
<!--
Build a test set of questions on varied public sites (static pages, JS-rendered pages, tables, pagination, cookie banners), including questions with **no answer** and **multi-page** questions. Fill in the results:

| Metric | Value |
|--------|-------|
| Exact-match accuracy | _TBD_ |
| Correct abstention rate | _TBD_ |
| Citation set match | _TBD_ |
| Median / p95 latency per question | _TBD_ |
| Startup time (model load) | _TBD_ |
-->
## Troubleshooting

- **Empty answers for everything:** check `/tmp/server.log`; the model may still be loading or failed to load.
- **`pip replaced ROCm torch`** during build: a dependency pulled in CUDA torch. Remove it from `requirements.txt` or install it with `--no-deps`.
- **`HTTP-only mode` in the log:** Chromium couldn't start. Static sites still work; JS-rendered sites will not.
- **Timeouts:** reduce `CTX_CHARS` and `max_new_tokens` in `server.py`.
