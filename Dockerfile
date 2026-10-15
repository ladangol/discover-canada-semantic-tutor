# Discover Canada Study. Built ON THE SERVER (x86_64): `deploy/deploy.sh` runs `docker build`.
# A container sidesteps the host's old Ubuntu/Python (3.6, glibc 2.27) and keeps the site's other apps untouched.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf TOKENIZERS_PARALLELISM=false STUDY_DATA_DIR=/data

RUN useradd --create-home --uid 1000 app && mkdir -p /data /opt/hf && chown -R app:app /data /opt/hf
WORKDIR /app

# CPU-only PyTorch first (the default wheel drags in ~2 GB of CUDA libraries), then the app's requirements.
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt

# Bake the MiniLM encoder into the image: no download at runtime, works even if Hugging Face is down.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" \
    && chown -R app:app /opt/hf

COPY --chown=app:app . .
USER app
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/discover_can_embedding_demo/_stcore/health', timeout=4).status == 200 else 1)"

CMD ["streamlit", "run", "study_app.py", \
     "--server.address", "0.0.0.0", "--server.port", "8501", \
     "--server.baseUrlPath", "discover_can_embedding_demo", \
     "--server.headless", "true", "--server.fileWatcherType", "none", \
     "--browser.gatherUsageStats", "false"]
