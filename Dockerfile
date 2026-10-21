# qwen-tts-server: an OpenAI-compatible speech endpoint for Qwen3-TTS
# CustomVoice on CPU. The model (~4.5 GB) is downloaded on first start into
# HF_HOME (mount a volume there).
FROM python:3.12.13-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/models

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 sox \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

RUN useradd --uid 1000 --create-home app && mkdir -p /models && chown app /models
WORKDIR /app
COPY server.py text.py batching.py ./
USER 1000
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
