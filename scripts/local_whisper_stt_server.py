#!/usr/bin/env python3
"""Persistent warm local faster-whisper (+ optional Moonshine/Gemini) HTTP daemon on port 8765 for Home Assistant STT fallback."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import socket
import threading
import time
import urllib.parse
import urllib.request
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_orig_getaddrinfo = socket.getaddrinfo


def _ipv4_preferred_getaddrinfo(
    host: str | bytes | None,
    port: str | bytes | int | None,
    family: int = 0,
    type: int = 0,
    proto: int = 0,
    flags: int = 0,
):
    """Prefer IPv4 (AF_INET) when resolving Google API hosts to prevent 4s IPv6 SYN timeouts on dual-stack LANs."""
    if family == 0 and isinstance(host, str) and "googleapis.com" in host:
        try:
            return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
        except OSError:
            pass
    return _orig_getaddrinfo(host, port, family, type, proto, flags)


socket.getaddrinfo = _ipv4_preferred_getaddrinfo

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
_LOGGER = logging.getLogger("local_whisper_stt")

HOST = os.environ.get("WHISPER_HOST", "")
PORT = int(os.environ.get("WHISPER_PORT", "8765"))
MODEL_NAME = os.environ.get("WHISPER_MODEL", "base")
LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "cs")
CPU_THREADS = int(os.environ.get("WHISPER_THREADS", "4"))
WHISPER_DIR = os.environ.get("WHISPER_CACHE_DIR", "/mnt/nvme/ai_models/hf_cache")
MOONSHINE_BASE_DIR = os.environ.get(
    "MOONSHINE_BASE_DIR",
    "/mnt/nvme/ai_models/hf_cache/hub/models--UsefulSensors--moonshine/snapshots/48b4e427b587bcf67797a5be706d6ddc4a298149/onnx/merged/base/float",
)
HERMES_ENV = Path.home() / ".hermes" / ".env"
STATE_FILE = Path.home() / ".hermes" / "cache" / "stt-venv" / "last_gemini_model"

GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
]
GEMINI_URL_TMPL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

CZECH_PROMPT = (
    "Rozsviť, zhasni, zapni, vypni, vytáhni, zatáhni, "
    "v obýváku, obývák, v zádveří, v kuchyni, v ložnici, v koupelně, na chodbě, "
    "v pracovně, v garáži, v šatně, žaluzie, světla, FVE, Deye, Loxone, denní spotřeba."
)

_DICTATION_PREFIX_RE = re.compile(
    r"(?i)^(?:vyplň|vypln|napiš|napis|zadej|vepiš|vepis|vyhledej\s+v\s+poli)\b"
)

_ROOM_PATTERN = (
    r"(?:v\s+obýváku|v\s+obyvaku|obývák|obyvak|v\s+zádveří|v\s+zadveri|zádveří|zadveri|"
    r"v\s+kuchyni|kuchyň|kuchyn|v\s+ložnici|v\s+loznici|ložnice|loznice|"
    r"v\s+koupelně|v\s+koupelne|koupelna|na\s+chodbě|na\s+chodbe|v\s+chodbě|chodba|"
    r"v\s+pracovně|v\s+pracovne|pracovna|v\s+garáži|v\s+garazi|garáž|garaz|"
    r"v\s+šatně|v\s+satne|šatna|satna|na\s+terase|terasa|v\s+sauně|sauna)"
)
_CLIPPED_ZHASNI_RE = re.compile(
    rf"(?i)^(?:zhasni|zhasný|zhasny|hasni|zasni|vlastně|vlastne|hlasitě|hlasite|asi|jasně|jasne|zase)\s+({_ROOM_PATTERN})\b(.*)$"
)
_CLIPPED_ROZSVIT_RE = re.compile(
    rf"(?i)^(?:rozsviť|rozsvit|rozsvědí|rozsvedi|rozsvěd|rozsved|sviť|svit|rozhlas|prosím\s+rozsviť|vejít|vejit|půjde|pujde|jít|jit|být|byt)\s+({_ROOM_PATTERN})\b(.*)$"
)


def normalize_czech_transcript(raw_text: str, is_dictation: bool = False) -> str:
    """Normalize Czech STT transcript in command mode and preserve multilingual search terms in dictation mode."""
    cleaned = (raw_text or "").strip().strip('"').strip("'").strip()
    if not cleaned:
        return ""
    if cleaned.endswith(".") and (" " in cleaned or len(cleaned) <= 14 or is_dictation):
        cleaned = cleaned[:-1].strip()
    if is_dictation or _DICTATION_PREFIX_RE.match(cleaned):
        return cleaned
    replacements = (
        ("Denis potřeba", "denní spotřeba"),
        ("denis potřeba", "denní spotřeba"),
        ("Rozsvědí ", "Rozsviť "),
        ("rozsvědí ", "rozsviť "),
        ("Rozsvěd ", "Rozsviť "),
        ("rozsvěd ", "rozsviť "),
        ("Zhasný ", "Zhasni "),
        ("zhasný ", "zhasni "),
        ("asi povídáku", "zhasni v obýváku"),
        ("Asi povídáku", "Zhasni v obýváku"),
        ("hlasitě v obýváku", "zhasni v obýváku"),
        ("Hlasitě v obýváku", "Zhasni v obýváku"),
        ("v povídáku", "v obýváku"),
        ("povídáku", "v obýváku"),
        ("vzár veří", "v zádveří"),
        ("zadveří", "zádveří"),
        ("vzádveří", "v zádveří"),
        ("jak se maž", "jak se máš"),
    )
    for src, dst in replacements:
        if src in cleaned:
            cleaned = cleaned.replace(src, dst)
    cleaned = re.sub(
        r"(?i)^(?:hasni|zasni)\s+",
        lambda m: "Zhasni " if m.group(0)[0].isupper() else "zhasni ",
        cleaned,
    )
    m_zhasni = _CLIPPED_ZHASNI_RE.match(cleaned)
    if m_zhasni:
        verb = "Zhasni" if cleaned[0].isupper() else "zhasni"
        cleaned = f"{verb} {m_zhasni.group(1)}{m_zhasni.group(2)}"
    else:
        m_rozsvit = _CLIPPED_ROZSVIT_RE.match(cleaned)
        if m_rozsvit:
            verb = "Rozsviť" if cleaned[0].isupper() else "rozsviť"
            cleaned = f"{verb} {m_rozsvit.group(1)}{m_rozsvit.group(2)}"
    return cleaned


_model_lock = threading.Lock()
_LOGGER.info("Loading faster-whisper %s (int8, cpu_threads=%d)...", MODEL_NAME, CPU_THREADS)
t0 = time.monotonic()
download_root = WHISPER_DIR if Path(WHISPER_DIR).exists() else None
MODEL = WhisperModel(
    MODEL_NAME,
    device="cpu",
    compute_type="int8",
    cpu_threads=CPU_THREADS,
    download_root=download_root,
)

MOONSHINE_MODEL = None
MOONSHINE_TOKENIZER = None
try:
    if Path(MOONSHINE_BASE_DIR).exists():
        from moonshine_onnx import MoonshineOnnxModel, load_tokenizer

        MOONSHINE_MODEL = MoonshineOnnxModel(
            models_dir=MOONSHINE_BASE_DIR, model_name="base", model_precision="float"
        )
        MOONSHINE_TOKENIZER = load_tokenizer()
        _LOGGER.info("Loaded Moonshine base ONNX for English fallback.")
except Exception as moon_err:
    _LOGGER.warning("Moonshine ONNX optional fallback not loaded: %s", moon_err)

# Warmup pass with 0.4s silent WAV so CTranslate2 threadpool is initialized immediately
_warmup_buf = io.BytesIO()
with wave.open(_warmup_buf, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(16000)
    wf.writeframes(b"\x00\x00" * 6400)
_warmup_buf.seek(0)
list(MODEL.transcribe(_warmup_buf, language=LANGUAGE, beam_size=1, temperature=0.0, without_timestamps=True))
_LOGGER.info("faster-whisper %s ready in %.2fs on port %d", MODEL_NAME, time.monotonic() - t0, PORT)


def _ensure_wav_bytes(raw_bytes: bytes) -> bytes:
    """Wrap raw PCM16 16kHz mono in a WAV container if not already RIFF WAV."""
    if raw_bytes.startswith(b"RIFF"):
        return raw_bytes
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(raw_bytes)
    return buf.getvalue()


def get_gemini_key() -> str:
    env_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if env_key:
        return env_key.strip()
    try:
        for line in HERMES_ENV.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", line)
            if m and m.group(1) in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
                val = m.group(2).strip().strip('"').strip("'")
                if val:
                    return val
    except Exception:
        pass
    return ""


def transcribe_online_gemini(wav_bytes: bytes, lang: str, is_dictation: bool = False) -> tuple[str, str]:
    key = get_gemini_key()
    if not key:
        raise RuntimeError("No GEMINI_API_KEY available")
    if is_dictation:
        prompt = (
            "Transcribe the spoken search query or dictated text from the audio clip accurately word-for-word. "
            "The speaker may dictate in Czech, English, or a mix of both. Do NOT translate English words into Czech phonetics. "
            "Return ONLY the raw transcription without quotes."
        )
    elif lang == "cs":
        prompt = "Přepiš tuto českou nahrávku. Přepis musí být plynulá, čistá česká věta bez opakování a bez fillerů. Piš jen přepis."
    else:
        prompt = "Transcribe this English audio clip cleanly and accurately. Output only the transcript."
    body = {
        "contents": [
            {
                "parts": [
                    {"inline_data": {"mime_type": "audio/wav", "data": base64.b64encode(wav_bytes).decode("ascii")}},
                    {"text": prompt},
                ]
            }
        ],
        "generationConfig": {"temperature": 0.0, "thinkingConfig": {"thinkingBudget": 0}},
    }
    payload = json.dumps(body).encode("utf-8")
    order = list(GEMINI_MODELS)
    try:
        last = STATE_FILE.read_text(encoding="utf-8").strip()
        if last in order:
            order.remove(last)
            order.insert(0, last)
    except Exception:
        pass

    errors = []
    for model in order:
        req = urllib.request.Request(
            GEMINI_URL_TMPL.format(model=model),
            data=payload,
            headers={"x-goog-api-key": key, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=7.5) as resp:
                data = json.loads(resp.read().decode("utf-8", "replace"))
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts).strip()
            if text:
                if lang == "cs" or is_dictation:
                    text = normalize_czech_transcript(text, is_dictation=is_dictation)
                try:
                    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
                    STATE_FILE.write_text(model, encoding="utf-8")
                except Exception:
                    pass
                return text, f"gemini:{model}"
        except Exception as exc:
            errors.append(f"{model}:{type(exc).__name__}")
            continue
    raise RuntimeError("; ".join(errors))


def wav_bytes_to_float32(raw_bytes: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(raw_bytes), "rb") as wf:
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        return pcm.astype(np.float32) / 32768.0


def transcribe_local_fallback(
    raw_bytes: bytes,
    lang: str = "cs",
    is_dictation: bool = False,
) -> tuple[str, str]:
    wav_bytes = _ensure_wav_bytes(raw_bytes)
    with _model_lock:
        if lang == "en" and not is_dictation and MOONSHINE_MODEL is not None and MOONSHINE_TOKENIZER is not None:
            audio_f32 = wav_bytes_to_float32(wav_bytes)
            tokens = MOONSHINE_MODEL.generate(audio_f32[np.newaxis, :])
            text = MOONSHINE_TOKENIZER.decode_batch(tokens)[0].strip()
            return text, "moonshine/base-onnx"

        whisper_lang = None if (is_dictation or lang == "auto") else (lang or LANGUAGE)
        whisper_prompt = None if (is_dictation or whisper_lang != "cs") else CZECH_PROMPT
        segments, _ = MODEL.transcribe(
            io.BytesIO(wav_bytes),
            language=whisper_lang,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            without_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={
                "threshold": 0.20,
                "min_speech_duration_ms": 80,
                "min_silence_duration_ms": 600,
                "speech_pad_ms": 800,
            },
            initial_prompt=whisper_prompt,
        )
        raw_text = " ".join(s.text.strip() for s in segments).strip()
        text = normalize_czech_transcript(raw_text, is_dictation=is_dictation)
        return text, f"faster-whisper-{MODEL_NAME}-int8"


class WhisperHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        return

    def _safe_write_json(self, status_code: int, data: dict[str, object]) -> None:
        resp = json.dumps(data, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
        except (BrokenPipeError, ConnectionResetError):
            _LOGGER.warning("Client disconnected before STT response could be written (BrokenPipe/ConnectionReset).")

    def do_GET(self) -> None:
        if self.path.startswith("/health"):
            self._safe_write_json(
                200,
                {
                    "status": "ok",
                    "primary_stt": "gemini-online",
                    "gemini_models": GEMINI_MODELS,
                    "fallback_cs": f"faster-whisper-{MODEL_NAME}-int8",
                    "fallback_en": "moonshine/base-onnx" if MOONSHINE_MODEL is not None else f"faster-whisper-{MODEL_NAME}-int8",
                },
            )
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if not parsed.path.startswith("/transcribe"):
            self.send_response(404)
            self.end_headers()
            return
        qs = urllib.parse.parse_qs(parsed.query)
        lang = (qs.get("lang", [LANGUAGE])[0] or LANGUAGE).strip().lower()
        mode = (qs.get("mode", ["auto"])[0] or "auto").strip().lower()
        dictation_flag = (qs.get("dictation", ["0"])[0] or "0").strip().lower() in ("1", "true", "yes")
        is_dictation = dictation_flag or lang == "auto"

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length > 0 else b""
        if not body:
            self._safe_write_json(400, {"error": "empty audio body", "text": ""})
            return

        t_start = time.perf_counter()
        try:
            if mode == "local":
                text, engine_used = transcribe_local_fallback(body, lang=lang, is_dictation=is_dictation)
            else:
                try:
                    text, engine_used = transcribe_online_gemini(body, lang=lang, is_dictation=is_dictation)
                except Exception as gem_err:
                    _LOGGER.warning("Online Gemini failed (%s), falling back to local (%s, dictation=%s)", gem_err, lang, is_dictation)
                    text, engine_used = transcribe_local_fallback(body, lang=lang, is_dictation=is_dictation)
            elapsed_ms = round((time.perf_counter() - t_start) * 1000.0, 1)
            _LOGGER.info(
                "Transcribed %d bytes in %.1fms (engine=%s, lang=%s, dictation=%s) -> %r",
                len(body),
                elapsed_ms,
                engine_used,
                lang,
                is_dictation,
                text,
            )
            self._safe_write_json(
                200,
                {
                    "text": text,
                    "engine": engine_used,
                    "language": lang,
                    "dictation": is_dictation,
                    "latency_ms": elapsed_ms,
                },
            )
        except Exception as err:
            _LOGGER.exception("Whisper transcribe error: %s", err)
            self._safe_write_json(500, {"error": str(err), "text": ""})


if __name__ == "__main__":
    ThreadingHTTPServer.daemon_threads = True
    server = ThreadingHTTPServer((HOST, PORT), WhisperHandler)
    server.serve_forever()
