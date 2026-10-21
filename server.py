"""qwen-tts-server: an OpenAI-compatible speech endpoint for Qwen3-TTS
CustomVoice on CPU.

POST /v1/audio/speech {"input", "voice", "response_format", "instructions",
"language"} answers WAV (16-bit mono, 24 kHz) or MP3; "instructions" ("Say
it angrily.") is Qwen's instruct (1.7B only; 0.6B ignores it), "language"
one of the model's ("English", "German", "Auto"; default LANGUAGE),
"speed" is not supported. The request shape is that of vLLM-Omni's Qwen3-TTS
endpoint, as audiobook-generator's engine of kind "qwen3" sends it.
GET /v1/audio/voices lists the model's speakers as Qwen names them ("Ryan",
"Ono_Anna"); any case is accepted, and a gender tag ("m_ryan") too.
GET /health answers 200 once the model is loaded (503 before). A model
that fails to load ends the process, so the kubelet starts it again (a
broken download is resumed).

Speed on CPU: decoding is bound by memory bandwidth (every generated frame
reads all weights), not by compute, so several texts generated together
cost little more than one. Every input is split into sentences, and the
sentences of all requests waiting (up to MAX_BATCH) are generated in one
batch; each request gets its sentences back in order, joined by a short
pause. Weights stay float32: bfloat16 has no native support on these CPUs
and int8 quantization made the speech unintelligible.
"""

import asyncio
import io
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from text import sentences

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
REVISION = os.environ.get("MODEL_REVISION") or None
THREADS = int(os.environ.get("THREADS", "0") or 0)
LANGUAGE = os.environ.get("LANGUAGE", "Auto")
MAX_BATCH = int(os.environ.get("MAX_BATCH", "16"))
# How long the batch waits for sentences of further requests.
BATCH_WAIT = float(os.environ.get("BATCH_WAIT_MS", "100")) / 1000
SENTENCE_PAUSE = float(os.environ.get("SENTENCE_PAUSE", "0.2"))
# Sentences shorter than this are joined with the next one (a lone
# "Yes." speaks oddly, and every row of a batch costs the same per frame).
MIN_CHARS = int(os.environ.get("MIN_CHARS", "40"))
# The 12Hz codec makes 12.5 frames per second of speech. Slow speech is
# about 10 characters per second; a sentence may take three times that
# before it is cut, so a run that does not end cannot grow without bound.
FRAMES_PER_CHAR = 12.5 / 10 * 3
MIN_FRAMES = 75
SAMPLE_RATE = 24000  # of the 12Hz codec's decoder

# Gender of Qwen3-TTS CustomVoice's speakers (model card).
GENDERS = {"aiden": "m", "dylan": "m", "eric": "m", "ryan": "m", "uncle_fu": "m",
           "ono_anna": "f", "serena": "f", "sohee": "f", "vivian": "f"}

log = logging.getLogger("qwen-tts")
logging.basicConfig(level=logging.INFO)
app = FastAPI(title="qwen-tts")
model = None
voice_names: list[str] = []  # as listed: "Ryan", "Ono_Anna"
speakers: dict[str, str] = {}  # accepted name (lower case) -> speaker
languages: set[str] = set()  # the model's (lower case); empty: any


@dataclass
class Piece:
    text: str
    speaker: str
    instruct: str
    language: str
    audio: np.ndarray | None = None
    error: Exception | None = None
    done: threading.Event = field(default_factory=threading.Event)


queue: list[Piece] = []
queued = threading.Condition()


def decode_each(decode):
    """qwen-tts decodes a batch's codes to audio in one call, every row
    padded to the longest: with 8 rows that took 3 GiB more memory.
    One row at a time costs about the same time and only its own length."""
    def wrapped(encoded):
        if not isinstance(encoded, list):
            return decode(encoded)
        wavs, sr = [], SAMPLE_RATE
        for e in encoded:
            w, sr = decode([e])
            wavs += w
        return wavs, sr
    return wrapped


def load():
    global model
    try:
        from huggingface_hub import snapshot_download
        from qwen_tts import Qwen3TTSModel

        if THREADS > 0:
            torch.set_num_threads(THREADS)
        # The whole repository at one revision: qwen-tts itself loads its
        # tokenizers from the default branch whatever revision it is given.
        path = MODEL if os.path.isdir(MODEL) else snapshot_download(MODEL, revision=REVISION)
        log.info("loading %s at %s", MODEL, REVISION or "main")
        m = Qwen3TTSModel.from_pretrained(path, device_map="cpu", dtype=torch.float32, attn_implementation="sdpa")
        m.model.speech_tokenizer.decode = decode_each(m.model.speech_tokenizer.decode)
        for s in m.get_supported_speakers() or []:
            voice_names.append("_".join(w.capitalize() for w in s.split("_")))
            speakers[s.lower()] = s
            if s in GENDERS:
                speakers[f"{GENDERS[s]}_{s}"] = s
        languages.update(x.lower() for x in m.get_supported_languages() or [])
        model = m
        log.info("model loaded, speakers %s, languages %s", voice_names, sorted(languages))
    except Exception:
        log.exception("loading the model failed")
        os._exit(1)
    worker()


def worker():
    while True:
        with queued:
            while not queue:
                queued.wait()
        # Let the sentences of requests arriving together join the batch.
        time.sleep(BATCH_WAIT)
        with queued:
            batch, queue[:] = queue[:MAX_BATCH], queue[MAX_BATCH:]
        started = time.perf_counter()
        try:
            frames = max(MIN_FRAMES, math.ceil(max(len(p.text) for p in batch) * FRAMES_PER_CHAR))
            with torch.inference_mode():
                wavs, _ = model.generate_custom_voice(
                    text=[p.text for p in batch], speaker=[p.speaker for p in batch],
                    instruct=[p.instruct for p in batch], language=[p.language for p in batch],
                    max_new_tokens=frames)
            for p, w in zip(batch, wavs):
                p.audio = np.asarray(w, dtype=np.float32).reshape(-1)
        except Exception as e:  # every request of the batch fails with it
            log.exception("generation failed")
            for p in batch:
                p.error = e
        seconds = sum(p.audio.size for p in batch if p.audio is not None) / SAMPLE_RATE
        log.info("batch of %d: %.1f s of speech in %.1f s", len(batch), seconds, time.perf_counter() - started)
        for p in batch:
            p.done.set()


threading.Thread(target=load, daemon=True).start()


class Speech(BaseModel):
    input: str = Field(min_length=1, max_length=4096)
    voice: str = Field(min_length=1, max_length=64)
    model: str = Field(default="", max_length=128)
    response_format: str = Field(default="wav", pattern="^(wav|mp3)$")
    instructions: str = Field(default="", max_length=500)
    language: str = Field(default="", max_length=40)
    speed: float | None = None  # not supported; the speaker's own pace


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(503, "loading the model")
    return {"status": "ok", "model": MODEL}


@app.get("/v1/audio/voices")
def voices():
    if model is None:
        raise HTTPException(503, "loading the model")
    return {"voices": sorted(voice_names)}


@app.post("/v1/audio/speech")
async def speech(req: Speech):
    if model is None:
        raise HTTPException(503, "loading the model")
    speaker = speakers.get(req.voice.removesuffix(".wav").lower())
    if speaker is None:
        raise HTTPException(400, f"unknown voice {req.voice!r}; see /v1/audio/voices")
    # Checked here, so a wrong one does not fail the batch it would join.
    language = req.language.strip() or LANGUAGE
    if languages and language.lower() not in languages | {"auto"}:
        raise HTTPException(400, f"unknown language {language!r}; one of {sorted(languages)}")
    pieces = [Piece(s, speaker, req.instructions.strip(), language) for s in sentences(req.input, MIN_CHARS)]
    if not pieces:
        raise HTTPException(400, "no text to speak")
    with queued:
        queue.extend(pieces)
        queued.notify()
    for p in pieces:
        await asyncio.to_thread(p.done.wait)
        if p.error is not None:
            raise HTTPException(500, f"generation failed: {p.error}")
    pause = np.zeros(int(SENTENCE_PAUSE * SAMPLE_RATE), dtype=np.float32)
    audio = np.concatenate([x for p in pieces for x in (p.audio, pause)][:-1])
    if audio.size == 0:
        raise HTTPException(500, "no audio generated")
    buf = io.BytesIO()
    if req.response_format == "mp3":
        sf.write(buf, audio, SAMPLE_RATE, format="MP3")
        return Response(buf.getvalue(), media_type="audio/mpeg")
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return Response(buf.getvalue(), media_type="audio/wav")
