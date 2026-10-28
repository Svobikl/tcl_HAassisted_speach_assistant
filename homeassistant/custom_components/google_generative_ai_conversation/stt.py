"""Speech-to-Text (STT) entity for Google Generative AI Conversation with Local Faster-Whisper Fallback."""

from __future__ import annotations

import array
import asyncio
from collections.abc import AsyncIterable
import io
import json
import logging
import math
import pathlib
import re
import socket
import sys
import time
import urllib.request
import wave

from google.genai import errors as genai_errors, types

_orig_getaddrinfo = socket.getaddrinfo


def _ipv4_preferred_getaddrinfo(
    host: str | bytes | None,
    port: str | bytes | int | None,
    family: int = 0,
    type: int = 0,
    proto: int = 0,
    flags: int = 0,
):
    """Prefer IPv4 (AF_INET) when resolving Google API hosts to prevent 4s IPv6 SYN blackholes on dual-stack LANs."""
    if family == 0 and isinstance(host, str) and "googleapis.com" in host:
        try:
            return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
        except OSError:
            pass
    return _orig_getaddrinfo(host, port, family, type, proto, flags)


socket.getaddrinfo = _ipv4_preferred_getaddrinfo
import voluptuous as vol

from homeassistant.components import stt
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import (
    CONF_PROMPT,
    DOMAIN,
    RECOMMENDED_STT_MODEL,
    RECOMMENDED_STT_OPTIONS,
)
from .entity import GoogleGenerativeAILLMBaseEntity

_LOGGER = logging.getLogger(__name__)

DEFAULT_STT_PROMPT = "Transcribe the spoken audio accurately."

CZECH_SMART_HOME_STT_PROMPT = (
    "Přepiš mluvenou řeč z nahrávky přesně slovo od slova. "
    "Vrať pouze samotný přepis řeči bez uvozovek a bez komentářů. "
    "Pokud nahrávka obsahuje pouze ticho nebo šum, vrať prázdný řetězec. "
    "Časté povely a místnosti v domě (mohou zaznít samostatně nebo ve spojení): "
    "rozsviť, zhasni, zapni, vypni, vytáhni, zatáhni, "
    "v obýváku, obývák, v zádveří, zádveří, v kuchyni, kuchyň, v ložnici, ložnice, "
    "v koupelně, koupelna, na chodbě, chodba, v pracovně, pracovna, v garáži, garáž, "
    "v šatně, šatna, na terase, terasa, v sauně, sauna, žaluzie, světla, "
    "FVE, Deye, Loxone, Eastron, denní spotřeba. "
    "Pokud příkaz začíná slovy 'vyplň do pole', 'napiš do pole', 'zadej do pole', "
    "'vepiš do pole', 'vyhledej v poli', 'vyplň' nebo 'napiš', může za ním následovat "
    "český nebo anglický hledaný výraz (názvy filmů, YouTube videí, jmen nebo anglických "
    "technických pojmů jako Matrix, Matt Armstrong, LLM inference on MacBook) — "
    "anglická slova nepřekládej do češtiny a zapiš je v původním anglickém pravopisu."
)

DICTATION_MULTILINGUAL_STT_PROMPT = (
    "Transcribe the spoken search query or dictated text from the audio clip accurately word-for-word. "
    "The speaker may dictate in Czech, English, or a mix of Czech and English (such as movie titles, "
    "YouTube search queries, channel names, people, products, or technical terms like 'Matrix', "
    "'Matt Armstrong', 'LLM inference on MacBook', 'Raspberry Pi 5', 'Toyota Yaris GR', 'Pelíšky'). "
    "Do NOT translate English words into Czech phonetics. "
    "Return ONLY the raw transcription without quotes, explanations, or trailing periods. "
    "If the audio contains only silence or noise, return an empty string."
)

FAST_STT_MODEL_CANDIDATES: tuple[str, ...] = (
    "models/gemini-2.5-flash",
    "models/gemini-3.6-flash",
    "models/gemini-3.5-flash",
    "models/gemini-3-flash-preview",
)

DEFAULT_GEMINI_STT_TIMEOUT_SEC = 5.5
DICTATION_GEMINI_STT_TIMEOUT_SEC = 7.5
MODEL_COOLDOWN_SEC = 10.0

DICTATION_MODE_FILE = pathlib.Path("/config/.tcl_tv_dictation_mode.json")
LOCAL_WHISPER_URL = "http://localhost:8765/transcribe?lang=cs&mode=local"

SILENCE_MARKERS: frozenset[str] = frozenset(
    {
        "<noise>",
        "<silence>",
        "[noise]",
        "[silence]",
        "(noise)",
        "(silence)",
        "noise",
        "silence",
    }
)

LOW_RMS_HALLUCINATIONS: frozenset[str] = frozenset(
    {
        "děkuji",
        "děkuji.",
        "díky",
        "titulky",
        "prosím",
    }
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
    rf"(?i)^(?:rozsviť|rozsvit|rozsvědí|rozsvedi|rozsvěd|rozsved|sviť|svit|rozhlas|prosím\s+rozsviť)\s+({_ROOM_PATTERN})\b(.*)$"
)

STT_SCHEMA = vol.Schema(
    {
        vol.Optional(CONF_PROMPT, default=DEFAULT_STT_PROMPT): cv.string,
    }
)

_preferred_working_stt_model: str = FAST_STT_MODEL_CANDIDATES[0]
_model_cooldown_until: dict[str, float] = {}


def _peek_dictation_mode(mode_file: pathlib.Path = DICTATION_MODE_FILE) -> bool:
    """Non-destructively check whether the long-press dictation mode flag file is active and not expired."""
    try:
        if not mode_file.exists():
            return False
        data = json.loads(mode_file.read_text(encoding="utf-8"))
        if not data.get("active"):
            return False
        expires_at = float(data.get("expires_at", 0.0))
        return time.time() <= expires_at
    except Exception:
        return False


def _compute_pcm16_rms(audio: bytes) -> float:
    """Compute root-mean-square (RMS) energy of 16-bit signed little-endian PCM."""
    if len(audio) < 2:
        return 0.0
    n_samples = len(audio) // 2
    samples = array.array("h")
    samples.frombytes(audio[: n_samples * 2])
    if sys.byteorder != "little":
        samples.byteswap()
    sum_sq = sum(s * s for s in samples)
    return math.sqrt(sum_sq / n_samples)


def _normalize_czech_stt_transcript(
    raw_text: str | None,
    audio_rms: float,
    is_dictation: bool = False,
) -> str:
    """Clean STT transcript, repair clipped Czech fricatives in command mode, and preserve multilingual queries in dictation mode."""
    if not raw_text:
        return ""
    cleaned = raw_text.strip().strip('"').strip("'").strip()
    if not cleaned:
        return ""
    lowered = cleaned.lower()
    if lowered in SILENCE_MARKERS:
        return ""
    if audio_rms < 190.0 and lowered in LOW_RMS_HALLUCINATIONS:
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


def _transcribe_with_local_whisper_sync(
    wav_bytes: bytes,
    lang: str = "cs",
    is_dictation: bool = False,
) -> str | None:
    """Call the warm local faster-whisper daemon on localhost:8765 in local-only mode."""
    if is_dictation:
        url = "http://localhost:8765/transcribe?lang=auto&mode=local&dictation=1"
    else:
        lang_code = "en" if (lang or "").lower().startswith("en") else "cs"
        url = f"http://localhost:8765/transcribe?lang={lang_code}&mode=local"
    req = urllib.request.Request(
        url,
        data=wav_bytes,
        headers={"Content-Type": "audio/wav"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=12.0) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data.get("text", "")


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up STT entities from config entry."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "stt":
            continue

        async_add_entities(
            [GoogleGenerativeAISttEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class GoogleGenerativeAISttEntity(stt.SpeechToTextEntity, GoogleGenerativeAILLMBaseEntity):
    """Google Generative AI Speech-to-Text entity with Local Faster-Whisper Fallback."""

    _attr_has_entity_name = False

    def __init__(
        self,
        entry: ConfigEntry,
        subentry: ConfigSubentry,
    ) -> None:
        """Initialize the STT entity."""
        super().__init__(entry, subentry, RECOMMENDED_STT_MODEL)

    @property
    def supported_languages(self) -> list[str]:
        """Return a list of supported languages."""
        return [
            "cs-CZ", "cs", "sk-SK", "sk", "en-US", "en-GB", "en", "de-DE", "de", "pl-PL", "pl",
            "ar", "bn", "bg", "zh", "hr", "da", "nl", "et",
            "fi", "fr", "el", "iw", "hi", "hu", "id", "it", "ja",
            "ko", "lv", "lt", "no", "pt", "ro", "ru", "sr",
            "sl", "es", "sw", "sv", "th", "tr", "uk", "vi",
        ]

    @property
    def supported_formats(self) -> list[stt.AudioFormats]:
        """Return a list of supported formats."""
        return [stt.AudioFormats.WAV, stt.AudioFormats.OGG]

    @property
    def supported_codecs(self) -> list[stt.AudioCodecs]:
        """Return a list of supported codecs."""
        return [stt.AudioCodecs.PCM, stt.AudioCodecs.OPUS]

    @property
    def supported_bit_rates(self) -> list[stt.AudioBitRates]:
        """Return a list of supported bitrates."""
        return [stt.AudioBitRates.BITRATE_16]

    @property
    def supported_sample_rates(self) -> list[stt.AudioSampleRates]:
        """Return a list of supported samplerates."""
        return [stt.AudioSampleRates.SAMPLERATE_16000]

    @property
    def supported_channels(self) -> list[stt.AudioChannels]:
        """Return a list of supported channels."""
        return [stt.AudioChannels.CHANNEL_MONO]

    async def async_process_audio_stream(
        self, metadata: stt.SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> stt.SpeechResult:
        """Process an audio stream to STT service with fast Gemini -> Local Whisper fallback."""
        global _preferred_working_stt_model
        t_start = time.monotonic()
        audio_chunks: list[bytes] = []
        async for chunk in stream:
            audio_chunks.append(chunk)

        raw_audio_bytes = b"".join(audio_chunks)
        if not raw_audio_bytes:
            return stt.SpeechResult("", stt.SpeechResultState.ERROR)

        if metadata.format == stt.AudioFormats.WAV:
            pcm_payload = (
                raw_audio_bytes[44:]
                if raw_audio_bytes.startswith(b"RIFF") and len(raw_audio_bytes) > 44
                else raw_audio_bytes
            )
            audio_rms = _compute_pcm16_rms(pcm_payload)
            if audio_rms < 55.0:
                _LOGGER.debug("Skipping STT on near-silent PCM buffer (RMS=%.1f)", audio_rms)
                return stt.SpeechResult("", stt.SpeechResultState.ERROR)

            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wf:
                wf.setnchannels(metadata.channel)
                wf.setsampwidth(metadata.bit_rate // 8)
                wf.setframerate(metadata.sample_rate)
                wf.writeframes(pcm_payload)
            audio_bytes = wav_buffer.getvalue()
            mime_type = "audio/wav"
        elif metadata.format == stt.AudioFormats.OGG:
            audio_rms = 500.0
            audio_bytes = raw_audio_bytes
            mime_type = "audio/ogg"
        else:
            return stt.SpeechResult("", stt.SpeechResultState.ERROR)

        is_dictation = _peek_dictation_mode(DICTATION_MODE_FILE)
        lang = (metadata.language or "").lower()
        if is_dictation:
            prompt = DICTATION_MULTILINGUAL_STT_PROMPT
            gemini_timeout = DICTATION_GEMINI_STT_TIMEOUT_SEC
        elif lang.startswith("cs"):
            prompt = CZECH_SMART_HOME_STT_PROMPT
            gemini_timeout = DEFAULT_GEMINI_STT_TIMEOUT_SEC
        else:
            prompt = self.subentry.data.get(
                CONF_PROMPT, RECOMMENDED_STT_OPTIONS[CONF_PROMPT]
            )
            gemini_timeout = DEFAULT_GEMINI_STT_TIMEOUT_SEC

        force_local_whisper = (
            pathlib.Path("/config/.force_local_whisper_stt").exists()
            or pathlib.Path("/tmp/force_local_whisper_stt").exists()
        )

        if not force_local_whisper:
            ordered_models: list[str] = [_preferred_working_stt_model]
            for m in FAST_STT_MODEL_CANDIDATES:
                if m not in ordered_models:
                    ordered_models.append(m)

            now = time.monotonic()
            active_candidates = [
                m for m in ordered_models if now >= _model_cooldown_until.get(m, 0.0)
            ]
            if not active_candidates:
                active_candidates = list(ordered_models)

            for candidate_model in active_candidates:
                try:
                    t_call = time.monotonic()
                    async with asyncio.timeout(gemini_timeout):
                        response = await self._genai_client.aio.models.generate_content(
                            model=candidate_model,
                            contents=[
                                prompt,
                                types.Part.from_bytes(
                                    data=audio_bytes,
                                    mime_type=mime_type,
                                ),
                            ],
                            config=types.GenerateContentConfig(
                                temperature=0.0,
                                max_output_tokens=96,
                                thinking_config=types.ThinkingConfig(thinking_budget=0),
                            ),
                        )
                    _preferred_working_stt_model = candidate_model
                    _model_cooldown_until.pop(candidate_model, None)
                    normalized = _normalize_czech_stt_transcript(
                        response.text, audio_rms, is_dictation=is_dictation
                    )
                    _LOGGER.info(
                        "Gemini STT (%s, dictation=%s) completed in %.2fs (total %.2fs, RMS=%.1f): %r",
                        candidate_model,
                        is_dictation,
                        time.monotonic() - t_call,
                        time.monotonic() - t_start,
                        audio_rms,
                        normalized,
                    )
                    if not normalized:
                        return stt.SpeechResult("", stt.SpeechResultState.ERROR)
                    return stt.SpeechResult(normalized, stt.SpeechResultState.SUCCESS)
                except (genai_errors.APIError, TimeoutError, Exception) as err:
                    _model_cooldown_until[candidate_model] = time.monotonic() + MODEL_COOLDOWN_SEC
                    _LOGGER.warning(
                        "Gemini STT model %s failed/exhausted (%s), trying next candidate or local Whisper...",
                        candidate_model,
                        err,
                    )

        # Fallback to Persistent Warm Local Faster-Whisper Daemon on RPi5 (localhost:8765)
        try:
            t_whisper = time.monotonic()
            whisper_text = await self.hass.async_add_executor_job(
                _transcribe_with_local_whisper_sync, audio_bytes, lang, is_dictation
            )
            normalized = _normalize_czech_stt_transcript(
                whisper_text, audio_rms, is_dictation=is_dictation
            )
            _LOGGER.info(
                "Local Faster-Whisper STT fallback (dictation=%s) completed in %.2fs (total %.2fs, RMS=%.1f): %r",
                is_dictation,
                time.monotonic() - t_whisper,
                time.monotonic() - t_start,
                audio_rms,
                normalized,
            )
            if not normalized:
                return stt.SpeechResult("", stt.SpeechResultState.ERROR)
            return stt.SpeechResult(normalized, stt.SpeechResultState.SUCCESS)
        except Exception as whisper_err:
            _LOGGER.error("Local Faster-Whisper STT fallback failed: %s", whisper_err)
            return stt.SpeechResult("", stt.SpeechResultState.ERROR)
