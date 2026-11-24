# Mandated ROCm base image (checked by layer identity). Do not change.
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

WORKDIR /app

# --- Python deps: NEVER let pip replace the ROCm torch -------------------------
COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
 && pip install --no-cache-dir --no-deps -U transformers tokenizers safetensors huggingface_hub \
 && python3 -c "import torch; print(torch.__version__); assert 'rocm' in torch.__version__, 'pip replaced ROCm torch!'" \
 && python3 -c "from transformers import AutoProcessor, AutoModelForImageTextToText; print('transformers ok')"

# --- Headless Chromium baked in (no network at evaluation time) ----------------
# Browsers go to a fixed path so they are found at runtime whatever the user/HOME.
# If Playwright does not support this Ubuntu release the build still succeeds and
# the agent falls back to plain HTTP fetching (see fetch_html in app code).
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
RUN (playwright install --with-deps chromium \
     && python3 -c "from playwright.sync_api import sync_playwright as s; p=s().start(); b=p.chromium.launch(); b.close(); p.stop(); print('chromium ok')") \
    || echo "WARNING: Chromium unavailable, running in HTTP-only mode"

# --- Weights shipped IN the image (reuse the same Qwen2.5-VL-7B) -----
# It handles both text extraction and images/screenshots, so one model is enough.
#   python3 -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen2.5-VL-7B-Instruct', local_dir='models/vlm')"
COPY models/ /models/
ENV MC3_MODEL_DIR=/models/vlm \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1

COPY app/ /app/
RUN mkdir -p /app/output

# Resident server: loads the model + warms Chromium ONCE (inside the startup budget),
# then the container stays alive. `app.py` is only a thin client talking to it.
CMD ["sh", "-c", "python3 /app/server.py >/tmp/server.log 2>&1 & exec sleep infinity"]
