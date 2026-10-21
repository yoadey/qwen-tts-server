# qwen-tts-server

An OpenAI-compatible speech server for
[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) CustomVoice on CPU, made for
reading books aloud (e.g. as an engine of
[audiobook-generator](https://github.com/yoadey/audiobook-generator)).
Emotions are given in words per request ("Say it angrily."). The request
shape is that of vLLM-Omni's Qwen3-TTS endpoint, so audiobook-generator's
engine of kind `qwen3` talks to it directly.

## API

| Endpoint | |
|---|---|
| `POST /v1/audio/speech` | `{"input", "voice", "instructions", "language", "response_format"}` → WAV (16-bit mono, 24 kHz, default) or MP3. `instructions` is Qwen's instruct (1.7B only; the 0.6B model ignores it). `language` is one of the model's (`English`, `German`, `Auto`, …; default `LANGUAGE`). `model` and `speed` are accepted and ignored. |
| `GET /v1/audio/voices` | The model's speakers as Qwen names them: `Aiden`, `Ryan`, `Serena`, `Ono_Anna`, … (any case is accepted, and a gender tag like `m_ryan` too) |
| `GET /health` | 200 once the model is loaded, 503 before |

```sh
curl -s localhost:8000/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input": "Then light the lamp! Light it now!", "voice": "Serena", "language": "English", "instructions": "Say it shouting."}' \
  -o out.wav
```

## How it is fast enough on a CPU

Decoding is bound by memory bandwidth: every generated frame reads all
weights, so several texts generated together cost little more than one.
Every input is split into sentences, and the sentences of all waiting
requests (up to `MAX_BATCH`) are generated in one batch; each request gets
its sentences back in order, joined by a short pause. Send several requests
at once (or long ones) to use it.

Measured on 4 vCPUs (Intel, AVX2/AVX-VNNI, DDR4) with 3 threads, 1.7B,
float32, an English passage of 10 lines (narration and dialogue with
emotions, ~60 s of speech), sent several times at once:

| Sentences in one batch | Real-time factor of the batch (lower is faster) | Peak memory | Whisper word error rate |
|---|---|---|---|
| 1 (one request after another) | 6.4 | – | 4.0 % |
| 8 | 2.3 | 9.0 GiB | 4.0 % |
| 16 (default) | 1.5 | 9.9 GiB | 2.9 % |
| 32 | 1.15 | 11.1 GiB | 2.6 % |

(The error rates are Whisper's mishearings, "boat" for "bolt" and the like;
batching does not cost quality.) A batch takes as long as its longest
sentence, and a request waits for all batches its sentences are in: with 32,
a batch took 3.5 minutes.

Weights stay float32. bfloat16 is slow on CPUs without native support, and
dynamic int8 quantization made the speech unintelligible (Whisper word error
rate 58–82 % against 0 % for float32).

Memory (1.7B): about 8 GiB after loading, plus the batch (table above;
real book sentences are longer than the benchmark's, and a batch's memory
grows with its rows times its longest sentence, which `MAX_ROW_FRAMES`
bounds; every batch logs its peak);
loading peaks briefly at about 11.7 GiB (partly the mapped weights file).
qwen-tts decodes a whole batch to audio at once, every row padded to the
longest, which took 3 GiB more with 8 rows; the server decodes one row at a
time instead. And qwen-tts's generate keeps the hidden states of all 29
layers for every frame of every row, though it reads only the last layer's;
the server keeps only that one.

## Configuration

| Variable | Default | |
|---|---|---|
| `MODEL` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | Hugging Face repository or a local directory (`…-0.6B-CustomVoice` is faster but ignores instructions) |
| `MODEL_REVISION` | `main` | Commit to download the whole repository at |
| `HF_HOME` | `/models` | Where the model is downloaded on first start (~4.5 GB); mount a volume |
| `HF_TOKEN` | | Only needed for rate limits |
| `THREADS` | torch's default | CPU threads; leave one core for the rest of the node |
| `LANGUAGE` | `Auto` | Language of requests that name none (`Auto`, `English`, `German`, …) |
| `MAX_BATCH` | `16` | Sentences generated together, at most |
| `MAX_ROW_FRAMES` | `6000` | Rows times the longest sentence's frame cap (2.5 frames per character, at least 75: half the usual pace before a sentence is cut) a batch may hold. The oldest waiting sentence always goes; the others closest to it in length join, since every row runs as long as the batch's longest |
| `BATCH_WAIT_MS` | `100` | How long a batch waits for sentences of further requests |
| `MIN_CHARS` | `40` | Shorter sentences are joined with the next |
| `SENTENCE_PAUSE` | `0.2` | Seconds of silence between sentences |

A model that fails to load ends the process, so Kubernetes restarts it (a
broken download is resumed).

## Image

`ghcr.io/yoadey/qwen-tts-server:<version>`, released from tags `v*.*.*`.

```sh
docker run -p 8000:8000 -v qwen-models:/models -e THREADS=3 ghcr.io/yoadey/qwen-tts-server:0.2.0
```

## Licence

Apache-2.0, as Qwen3-TTS.
