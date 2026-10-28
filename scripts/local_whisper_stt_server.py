#!/usr/bin/env python3
"""Persistent warm local faster-whisper HTTP daemon on localhost:8765 for Home Assistant STT fallback."""

from __future__ import annotations

import io
import json
import logging
import os
import re
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
_LOGGER = logging.getLogger("local_whisper_stt")

HOST = os.environ.get("WHISPER_HOST", "localhost")
PORT = int(os.environ.get("WHISPER_PORT", "8765"))
MODEL_NAME = os.environ.get("WHISPER_MODEL", "base")
LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "cs")
CPU_THREADS = int(os.environ.get("WHISPER_THREADS", "4"))

CZECH_PROMPT = (
    "Rozsviť, zhasni, zapni, vypni, vytáhni, zatáhni, "
    "v obýváku, obývák, v zádveří, v kuchyni, v ložnici, v koupelně, na chodbě, "
    "v pracovně, v garáži, v šatně, žaluzie, světla, FVE, Deye, Loxone, denní spotřeba."
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
    rf"(?i)^(?:rozsviť|rozsvit|rozsvědí|rozsvedi|rozsvěd|rozsved|sviť|svit|rozhlas|prosím\s+rozsviť)\s+({_ROOM_PATTERN})\b(.*)$"
)


def normalize_czech_transcript(raw_text: str) -> str:
    """Normalize Czech STT transcript and repair clipped initial fricatives."""
    cleaned = (raw_text or "").strip()
    if not cleaned:
        return ""
    if cleaned.endswith(".") and (" " in cleaned or len(cleaned) <= 14):
        cleaned = cleaned[:-1].strip()
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
MODEL = WhisperModel(
    MODEL_NAME,
    device="cpu",
    compute_type="int8",
    cpu_threads=CPU_THREADS,
)
# Warmup pass with 0.4s silent WAV so CTranslate2 threadpool is initialized immediately
_warmup_buf = io.BytesIO()
with wave.open(_warmup_buf, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(16000)
    wf.writeframes(b"\x00\x00" * 6400)
_warmup_buf.seek(0)
list(MODEL.transcribe(_warmup_buf, language=LANGUAGE, beam_size=1))
_LOGGER.info("faster-whisper %s ready in %.2fs on http://%s:%d", MODEL_NAME, time.monotonic() - t0, HOST, PORT)


def _ensure_wav_bytes(raw_bytes: bytes) -> io.BytesIO:
    """Wrap raw PCM16 16kHz mono in a WAV container if not already RIFF WAV."""
    if raw_bytes.startswith(b"RIFF"):
        return io.BytesIO(raw_bytes)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(raw_bytes)
    buf.seek(0)
    return buf


class WhisperHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        if self.path.startswith("/health"):
            payload = json.dumps({"status": "ok", "model": f"faster-whisper-{MODEL_NAME}-int8"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        if not self.path.startswith("/transcribe"):
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length > 0 else b""
        if not body:
            self.send_response(400)
            self.end_headers()
            return
        t_start = time.monotonic()
        try:
            wav_io = _ensure_wav_bytes(body)
            with _model_lock:
                segments, _ = MODEL.transcribe(
                    wav_io,
                    language=LANGUAGE,
                    beam_size=1,
                    best_of=1,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    vad_filter=True,
                    vad_parameters={
                        "threshold": 0.20,
                        "min_speech_duration_ms": 80,
                        "min_silence_duration_ms": 600,
                        "speech_pad_ms": 800,
                    },
                    initial_prompt=CZECH_PROMPT,
                )
                text = normalize_czech_transcript(" ".join(s.text.strip() for s in segments).strip())
            elapsed = round(time.monotonic() - t_start, 3)
            _LOGGER.info("Transcribed %d bytes in %.3fs -> %r", len(body), elapsed, text)
            resp = json.dumps({"text": text, "elapsed_sec": elapsed}, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
        except Exception as err:
            _LOGGER.exception("Whisper transcribe error: %s", err)
            resp = json.dumps({"error": str(err), "text": ""}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), WhisperHandler)
    server.serve_forever()
