FROM python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/app/.cache/huggingface \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system app \
    && useradd --system --gid app --create-home --home-dir /home/app app \
    && mkdir -p /home/app/.cache/huggingface /app \
    && chown -R app:app /home/app /app

WORKDIR /app

COPY requirements.txt ./
# sentence-transformers needs PyTorch. Install its CPU build first so the
# resolver does not pull hundreds of megabytes of CUDA libraries into this image.
RUN python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.13.0" \
    && python -m pip install -r requirements.txt

COPY --chown=app:app . .
USER app

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).read()" || exit 1

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
