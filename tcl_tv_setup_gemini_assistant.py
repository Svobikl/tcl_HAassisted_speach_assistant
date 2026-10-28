#!/usr/bin/env python3
"""Automated TCL 65C815 Android TV Home Assistant + Gemini Voice Assistant Provisioner, Remote Hook & Energy VAD."""

from __future__ import annotations

import array
from collections import deque
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import sys
import threading
import time
import unicodedata
import zlib
from pathlib import Path

TV_TARGET = os.environ.get("TCL_TV_ADB_TARGET", "<TV_IP_ADDRESS>:5555")
APK_PATH = Path(os.environ.get("HA_APK_PATH", "/usr/share/hassio/homeassistant/www/ha-companion-tv.apk"))
STATE_FILE = Path(os.environ.get("TCL_STATE_FILE", "/usr/share/hassio/homeassistant/.tcl_tv_assistant_provisioned.json"))
DICTATION_MODE_FILE = Path(
    os.environ.get("TCL_DICTATION_MODE_FILE", "/usr/share/hassio/homeassistant/.tcl_tv_dictation_mode.json")
)
DICTATION_REQUEST_FILE = Path(
    os.environ.get("TCL_DICTATION_REQUEST_FILE", "/usr/share/hassio/homeassistant/.tcl_tv_type_request.json")
)
LONG_PRESS_HOLD_SEC = 0.65
DICTATION_TTL_SEC = 18.0
HA_PKG = "io.homeassistant.companion.android"
KATNISS_PKG = "com.google.android.katniss"
ASSIST_ACTIVITY = "io.homeassistant.companion.android/io.homeassistant.companion.android.assist.AssistActivity"
ASSIST_TASK_AFFINITY = "io.homeassistant.companion.android.assist"
VOICE_INTERACTION_SERVICE = (
    "io.homeassistant.companion.android/io.homeassistant.companion.android.assist.service.AssistVoiceInteractionService"
)
TCL_IME = "com.tcl.inputmethod.international/.T_IME"
REMOTE_BRIDGE_IME = "com.google.android.tv.remote.service/.ImeBridgeService"

# Do NOT include delayed SEARCH_LONG_PRESS broadcasts, which fire ~2.7s into an utterance
# and would re-trigger AssistActivity mid-recording.
REMOTE_BUTTON_LOG_TRIGGERS: tuple[str, ...] = (
    "act=android.speech.action.WEB_SEARCH",
    "act=android.speech.action.VOICE_SEARCH_HANDS_FREE",
    "OnConversationTurnStarted",
    "Mic button press count:",
    "SearchActivityTrampoline",
)

REMOTE_GETEVENT_KEY_TRIGGERS: tuple[str, ...] = (
    "KEY_SEARCH",
    "KEY_VOICE_ASSIST",
    "0246",
    "00d9",
)

DICTATION_LONG_PRESS_LOG_TRIGGERS: tuple[str, ...] = (
    "act=android.intent.action.SEARCH_LONG_PRESS",
)

_DICTATION_SUBMIT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?i)^(?:vyplň|vypln|napiš|napis|zadej|vepiš|vepis)\s+(?:do\s+pole\s+|do\s+políčka\s+|text\s+)?a\s+(?:vyhledej|hledej|potvrď|potvrd|odešli|odesli)\s*[:,-]?\s*(.+)$"
    ),
    re.compile(
        r"(?i)^(?:vyhledej|hledej)\s+(?:v\s+poli|do\s+pole|v\s+políčku|do\s+políčka)\s*[:,-]?\s*(.+)$"
    ),
)

_DICTATION_TYPE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?i)^(?:vyplň|vypln|napiš|napis|zadej|vepiš|vepis)(?:\s+(?:do\s+pole|do\s+políčka|do\s+vyhledávání|pole|políčko|text))?(?:\s*[:,-]\s*|\s+)(.*)$"
    ),
    re.compile(
        r"(?i)^(?:do\s+pole|do\s+políčka|do\s+vyhledávání)\s+(?:vyplň|vypln|napiš|napis|zadej|vepiš|vepis)(?:\s*[:,-]\s*|\s+)(.*)$"
    ),
)

_EMPTY_DICTATION_TARGETS: frozenset[str] = frozenset(
    {
        "do pole",
        "do políčka",
        "do policka",
        "do vyhledávání",
        "do vyhledavani",
        "pole",
        "políčko",
        "policko",
        "text",
    }
)

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

BARE_AREA_LIGHT_MAP: tuple[tuple[tuple[str, ...], str, tuple[str, ...]], ...] = (
    (
        ("v obýváku", "v obyvaku", "obývák", "obyvak", "v obývacím pokoji", "obývací pokoj"),
        "v obýváku",
        ("light.ovladani_osvetleni_kuchyne",),
    ),
    (
        ("v kuchyni", "kuchyň", "kuchyne", "kuchyni", "kuchyně"),
        "v kuchyni",
        ("light.ovladani_osvetleni_kuchyne",),
    ),
    (
        ("v zádveří", "v zadveri", "zádveří", "zadveri"),
        "v zádveří",
        ("light.zadveri", "light.ovladani_osvetleni_zadveri"),
    ),
    (
        ("v ložnici", "v loznici", "ložnice", "loznice"),
        "v ložnici",
        ("light.loznice", "light.ovladani_osvetleni_loznice"),
    ),
    (
        ("v koupelně", "v koupelne", "koupelna", "koupelně"),
        "v koupelně",
        ("light.koupelna", "light.ovladani_osvetleni_koupelna"),
    ),
    (
        ("na chodbě", "na chodbe", "v chodbě", "chodba", "chodbě"),
        "na chodbě",
        ("light.chodba", "light.ovladani_osvetleni_chodba"),
    ),
    (
        ("v pracovně", "v pracovne", "pracovna", "pracovně"),
        "v pracovně",
        ("light.pracovna", "light.ovladani_osvetleni_pracovna"),
    ),
    (
        ("v garáži", "v garazi", "garáž", "garaz", "garáži"),
        "v garáži",
        ("light.garaz", "light.ovladani_osvetleni_garaz"),
    ),
    (
        ("v šatně", "v satne", "šatna", "satna", "šatně"),
        "v šatně",
        ("light.satna", "light.ovladani_osvetleni_satna"),
    ),
)

# 2.5s debounce deduplicates simultaneous getevent + logcat triggers from a single button press
# while allowing rapid re-presses when the user starts a new command.
MIN_DEBOUNCE_SEC = 2.5

# 0.45s settle delay after `am stack remove` ensures the old AssistActivity's onDestroy() finishes
# sending `unsubscribe_events` on the shared WebSocketCoreImpl before the new AssistActivity opens
# its own assist_pipeline/run subscription (preventing a 30s WebSocketCoreImpl eventProcessor deadlock).
STACK_REMOVE_SETTLE_SEC = 0.45

# 1600 shorts (3200 bytes = 100ms @ 16kHz, 10 chunks/sec) prevents WebSocket/SQLite queue buildup
# in AssistViewModelBase$setupRecorder$1 (~51ms per chunk on TCL Android TV) so audio streams at 1.0x real-time.
VOICE_RECORDER_BUFFER_SHORTS = 1600

os.environ.setdefault("HOME", "/root")

_trigger_lock = threading.Lock()
_last_trigger_ts = 0.0


def compute_pcm16_rms(audio: bytes) -> float:
    """Compute root-mean-square (RMS) acoustic energy of 16-bit signed little-endian mono PCM."""
    if len(audio) < 2:
        return 0.0
    n_samples = len(audio) // 2
    samples = array.array("h")
    samples.frombytes(audio[: n_samples * 2])
    if sys.byteorder != "little":
        samples.byteswap()
    sum_sq = sum(s * s for s in samples)
    return math.sqrt(sum_sq / n_samples)


def normalize_czech_stt_transcript(raw_text: str | None, audio_rms: float = 500.0) -> str:
    """Normalize Czech STT output, repair clipped initial fricatives, and filter silence/noise markers."""
    if not raw_text:
        return ""
    cleaned = raw_text.strip()
    if not cleaned:
        return ""
    lowered = cleaned.lower()
    if lowered in SILENCE_MARKERS:
        return ""
    if audio_rms < 190.0 and lowered in LOW_RMS_HALLUCINATIONS:
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


def expand_bare_area_command(user_text: str, light_states: dict[str, str] | None = None) -> str:
    """Expand a bare Czech room name ('v obýváku') into an explicit state-aware light toggle command."""
    if not user_text:
        return user_text
    states = light_states or {}
    normalized = user_text.strip().rstrip(".!?,").strip().lower()
    for aliases, canonical_phrase, entity_ids in BARE_AREA_LIGHT_MAP:
        if normalized in aliases:
            any_on = any(states.get(eid) == "on" for eid in entity_ids)
            verb = "Zhasni" if any_on else "Rozsviť"
            return f"{verb} světla {canonical_phrase}"
    return user_text


def parse_dictation_command(user_text: str | None) -> tuple[str, bool] | None:
    """Parse Czech voice dictation commands ('Vyplň do pole: Matrix') into (dictated_text, submit_flag)."""
    if not user_text:
        return None
    cleaned = user_text.strip()
    if not cleaned:
        return None
    for pattern in _DICTATION_SUBMIT_PATTERNS:
        m = pattern.match(cleaned)
        if m:
            payload = m.group(1).strip().lstrip(":,-").rstrip(".!?").strip()
            if payload and payload.lower() not in _EMPTY_DICTATION_TARGETS:
                return (payload, True)
            return None
    for pattern in _DICTATION_TYPE_PATTERNS:
        m = pattern.match(cleaned)
        if m:
            payload = m.group(1).strip().lstrip(":,-").rstrip(".!?").strip()
            if payload and payload.lower() not in _EMPTY_DICTATION_TARGETS:
                return (payload, False)
            return None
    return None


def format_adb_input_text(text: str | None) -> str:
    """Transliterate Czech diacritics to ASCII, strip shell metacharacters, and encode spaces as %s for `adb shell input text`."""
    if not text:
        return ""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    safe_chars = re.sub(r"[^A-Za-z0-9 ._:@/+,-]", "", ascii_text)
    collapsed = re.sub(r"\s+", " ", safe_chars).strip()
    if not collapsed:
        return ""
    return collapsed.replace(" ", "%s")


def should_activate_dictation_from_log_line(line: str) -> bool:
    """Return True if a logcat line indicates a long-press on the TCL remote microphone button (SEARCH_LONG_PRESS)."""
    if not line:
        return False
    return any(token in line for token in DICTATION_LONG_PRESS_LOG_TRIGGERS)


def is_getevent_search_key_up(line: str) -> bool:
    """Return True if a kernel getevent -l line indicates the TCL_RC602S voice button was released (UP)."""
    if not line or "EV_KEY" not in line or "UP" not in line:
        return False
    return any(key in line for key in REMOTE_GETEVENT_KEY_TRIGGERS)


def activate_dictation_mode(
    source: str,
    mode_file: Path = DICTATION_MODE_FILE,
    ttl_sec: float = DICTATION_TTL_SEC,
) -> None:
    """Write the one-shot dictation mode flag file so the current Assist voice turn types raw text into the TV."""
    now = time.time()
    payload = {
        "active": True,
        "source": source,
        "ts": now,
        "expires_at": now + ttl_sec,
    }
    mode_file.parent.mkdir(parents=True, exist_ok=True)
    mode_file.write_text(json.dumps(payload), encoding="utf-8")


def clear_dictation_mode(mode_file: Path = DICTATION_MODE_FILE) -> None:
    """Remove any active dictation mode flag file."""
    try:
        if mode_file.exists():
            mode_file.unlink()
    except OSError:
        pass


def consume_dictation_mode(mode_file: Path = DICTATION_MODE_FILE) -> bool:
    """Atomically consume and delete the dictation mode flag file; returns True if active and not expired."""
    try:
        if not mode_file.exists():
            return False
        raw = mode_file.read_text(encoding="utf-8")
        mode_file.unlink(missing_ok=True)
        data = json.loads(raw)
        if not data.get("active"):
            return False
        expires_at = float(data.get("expires_at", 0.0))
        return time.time() <= expires_at
    except Exception:
        try:
            mode_file.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def enqueue_dictation_request(
    text: str,
    submit: bool = False,
    req_file: Path = DICTATION_REQUEST_FILE,
) -> None:
    """Write a dictation text injection request for the host ADB worker thread."""
    payload = {
        "text": text,
        "submit": bool(submit),
        "ts": time.time(),
    }
    req_file.parent.mkdir(parents=True, exist_ok=True)
    req_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def consume_dictation_request(req_file: Path = DICTATION_REQUEST_FILE) -> dict[str, object] | None:
    """Atomically read and remove a pending dictation request file if present."""
    try:
        if not req_file.exists():
            return None
        raw = req_file.read_text(encoding="utf-8")
        req_file.unlink(missing_ok=True)
        data = json.loads(raw)
        if isinstance(data, dict) and data.get("text"):
            return data
        return None
    except Exception:
        try:
            req_file.unlink(missing_ok=True)
        except OSError:
            pass
        return None


def simulate_vad_preroll_and_onset(
    total_pre_speech_chunks: int = 80,
    ble_pop_chunk_index: int = 40,
    speech_chunks_count: int = 60,
    post_silence_chunks_count: int = 50,
) -> dict[str, object]:
    """Simulate 500ms pre-roll ring buffer + 5-chunk sustained onset gate with a 10ms BLE startup pop."""
    pre_roll_buffer: deque[int] = deque(maxlen=50)
    yielded_chunk_indices: list[int] = []
    consecutive_onset_chunks = 0
    in_speech = False
    vad_start_chunk_index: int | None = None
    false_trigger_on_ble_pop = False

    total_chunks = total_pre_speech_chunks + speech_chunks_count + post_silence_chunks_count
    for idx in range(total_chunks):
        if idx == ble_pop_chunk_index:
            is_onset_candidate = True
        elif total_pre_speech_chunks <= idx < total_pre_speech_chunks + speech_chunks_count:
            is_onset_candidate = True
        else:
            is_onset_candidate = False

        if not in_speech:
            if is_onset_candidate:
                consecutive_onset_chunks += 1
            else:
                consecutive_onset_chunks = max(0, consecutive_onset_chunks - 1)
            if consecutive_onset_chunks >= 5:
                in_speech = True
                vad_start_chunk_index = idx
                if idx < total_pre_speech_chunks:
                    false_trigger_on_ble_pop = True
                while pre_roll_buffer:
                    yielded_chunk_indices.append(pre_roll_buffer.popleft())
                yielded_chunk_indices.append(idx)
            else:
                pre_roll_buffer.append(idx)
        else:
            yielded_chunk_indices.append(idx)

    return {
        "false_trigger_on_ble_pop": false_trigger_on_ble_pop,
        "vad_triggered": in_speech,
        "vad_start_chunk_index": vad_start_chunk_index,
        "preroll_chunks_preserved": (
            (vad_start_chunk_index - yielded_chunk_indices[0])
            if vad_start_chunk_index is not None and yielded_chunk_indices
            else 0
        ),
        "ble_pop_discarded_from_preroll": ble_pop_chunk_index < (yielded_chunk_indices[0] if yielded_chunk_indices else 0)
        or (
            vad_start_chunk_index is not None
            and (vad_start_chunk_index - ble_pop_chunk_index) <= 50
            and not false_trigger_on_ble_pop
        ),
    }


def patch_voice_recorder_buffer_shorts(
    dex_bytes: bytes, target_shorts: int = VOICE_RECORDER_BUFFER_SHORTS
) -> bytes:
    """Patch VoiceAudioRecorder 16kHz buffer size from 160 shorts (10ms) to target_shorts (1600 shorts = 100ms) and update DEX checksums."""
    if len(dex_bytes) < 32 or not dex_bytes.startswith(b"dex\n"):
        raise ValueError("Invalid DEX header magic")
    if target_shorts < 160 or target_shorts > 16000 or target_shorts % 160 != 0:
        raise ValueError(f"target_shorts must be a multiple of 160 in [160, 16000], got {target_shorts}")

    imm_bytes = struct.pack("<h", target_shorts)
    site1_orig = bytes.fromhex("1301803e1302a000")
    site1_new = bytes.fromhex("1301803e1302") + imm_bytes
    site2_orig = bytes.fromhex("1308b90128031308a000")
    site2_new = bytes.fromhex("1308b90128031308") + imm_bytes

    buf = bytearray(dex_bytes)
    pos1 = buf.find(site1_orig)
    if pos1 != -1:
        buf[pos1 : pos1 + len(site1_orig)] = site1_new
    elif buf.find(site1_new) == -1:
        raise ValueError("VoiceAudioRecorderKt.createVoiceAudioRecord patch site not found")

    pos2 = buf.find(site2_orig)
    if pos2 != -1:
        buf[pos2 : pos2 + len(site2_orig)] = site2_new
    elif buf.find(site2_new) == -1:
        raise ValueError("VoiceAudioRecorder$createSharedFlow$upstream$1 patch site not found")

    buf[12:32] = hashlib.sha1(buf[32:]).digest()
    struct.pack_into("<I", buf, 8, zlib.adler32(buf[12:]) & 0xFFFFFFFF)
    return bytes(buf)


class EnergySilenceDetector:
    """Pre-AGC RMS acoustic energy envelope + adaptive noise-floor silence detector for TCL_RC602S BLE mic."""

    def __init__(
        self,
        initial_noise_floor: float = 120.0,
        silence_floor_multiplier: float = 1.45,
        silence_peak_ratio: float = 0.065,
        max_silence_threshold: float = 245.0,
        min_speech_chunks_before_silence_gate: int = 25,
        min_consecutive_low_rms_chunks: int = 24,
    ) -> None:
        self.min_raw_rms = initial_noise_floor
        self.silence_floor_multiplier = silence_floor_multiplier
        self.silence_peak_ratio = silence_peak_ratio
        self.max_silence_threshold = max_silence_threshold
        self.min_speech_chunks_before_silence_gate = min_speech_chunks_before_silence_gate
        self.min_consecutive_low_rms_chunks = min_consecutive_low_rms_chunks
        self.ema_raw_rms = initial_noise_floor
        self.peak_raw_rms = 0.0
        self.speech_chunks = 0
        self.consecutive_low_rms_chunks = 0
        self.chunks_seen = 0

    def reset(self) -> None:
        """Reset state for a new voice command turn."""
        self.peak_raw_rms = 0.0
        self.speech_chunks = 0
        self.consecutive_low_rms_chunks = 0
        self.chunks_seen = 0

    def process_chunk(self, raw_audio: bytes, raw_vad_prob: float) -> float:
        """Combine raw MicroVad probability with pre-AGC RMS energy onset & silence detection."""
        raw_rms = compute_pcm16_rms(raw_audio)
        self.chunks_seen += 1
        if self.chunks_seen == 1:
            self.ema_raw_rms = raw_rms
            self.min_raw_rms = max(30.0, min(raw_rms, 220.0))
        else:
            self.ema_raw_rms = 0.35 * raw_rms + 0.65 * self.ema_raw_rms
            if self.ema_raw_rms < self.min_raw_rms:
                self.min_raw_rms = max(30.0, 0.85 * self.min_raw_rms + 0.15 * self.ema_raw_rms)
            else:
                self.min_raw_rms = min(220.0, 0.995 * self.min_raw_rms + 0.005 * self.ema_raw_rms)

        capped_ema = min(self.ema_raw_rms, 3500.0)
        onset_threshold = max(165.0, min(320.0, self.min_raw_rms * 1.45))

        # Before speech onset, reject idle BLE background noise even if Speex AGC boosts MicroVad probability
        if self.speech_chunks == 0:
            if max(raw_rms, self.ema_raw_rms) < onset_threshold or raw_vad_prob <= 0.35:
                return 0.0
            self.speech_chunks = 1
            self.consecutive_low_rms_chunks = 0
            self.peak_raw_rms = max(capped_ema, min(raw_rms, 3500.0))
            return max(raw_vad_prob, 0.85)

        if capped_ema > self.peak_raw_rms:
            self.peak_raw_rms = capped_ema

        silence_threshold = min(
            self.max_silence_threshold,
            max(
                150.0,
                self.min_raw_rms * self.silence_floor_multiplier,
                self.peak_raw_rms * self.silence_peak_ratio,
            ),
        )

        # Active speech energy present: keep probability >= 0.85 even during unvoiced consonant dips
        if self.ema_raw_rms >= silence_threshold:
            self.consecutive_low_rms_chunks = 0
            self.speech_chunks += 1
            return max(raw_vad_prob, 0.85)

        # Low RMS energy after speech onset: allow brief inter-word pauses, then drop to 0.0
        if self.speech_chunks < self.min_speech_chunks_before_silence_gate:
            self.speech_chunks += 1
            return max(raw_vad_prob, 0.80)

        self.consecutive_low_rms_chunks += 1
        if self.consecutive_low_rms_chunks >= self.min_consecutive_low_rms_chunks:
            return 0.0
        return max(raw_vad_prob, 0.80)


def build_immediate_mic_stop_stt_end_payload() -> dict[str, dict[str, object]]:
    """Build preliminary stt-end event data that stops TV recording immediately at stt-vad-end without adding an empty chat bubble."""
    return {"stt_output": {}}


SECRET_SCAN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("Google API Key", re.compile(r"AIza[0-9A-Za-z\-_]{20,}")),
    ("GitHub Token", re.compile(r"gh[pousr]_[0-9A-Za-z]{20,}")),
    ("JWT Token", re.compile(r"eyJ[0-9A-Za-z\-_]{10,}\.[0-9A-Za-z\-_]{10,}\.[0-9A-Za-z\-_]{10,}")),
    ("Private Key Block", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("IPv4 Address", re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")),
)


def scan_directory_for_secrets(root_dir: Path) -> list[str]:
    """Scan all text files under root_dir (excluding .git and __pycache__) for accidental API keys, tokens, IPs, or personal identifiers."""
    findings: list[str] = []
    for path in sorted(Path(root_dir).rglob("*")):
        if not path.is_file():
            continue
        if any(part in {".git", "__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for label, pattern in SECRET_SCAN_PATTERNS:
            for match in pattern.finditer(text):
                findings.append(f"{path}: {label} ({match.group(0)[:10]}...)")
    return findings


def format_smarthome_telemetry_summary(
    deye_attrs: dict[str, object] | None,
    extra_states: dict[str, str] | None = None,
) -> str:
    """Format live Smart Home Deye 15kW + house telemetry with explicit daily (kWh) and instant (W) digits."""
    attrs = deye_attrs or {}
    states = extra_states or {}

    def _val(key: str, unit: str = "") -> str:
        v = attrs.get(key)
        if v is None or v == "":
            return "neznámé"
        return f"{v} {unit}".strip() if unit else str(v)

    lines = [
        "ŽIVÁ TELEMETRIE CHYTRÉ DOMÁCNOSTI (aktuální stav právě teď — při odpovědi čti čísla přesně podle těchto číslic, nezaměňuj stovky za desítky):",
        f"- Aktuální okamžitý příkon (spotřeba) domu právě teď: {_val('load_total_w', 'W')} (L1: {_val('load_l1_w', 'W')}, L2: {_val('load_l2_w', 'W')}, L3: {_val('load_l3_w', 'W')})",
        f"- Denní spotřeba domu dnes (od půlnoci): {_val('day_load_kwh', 'kWh')} (celková kumulativní spotřeba: {_val('total_load_kwh', 'kWh')})",
        f"- Okamžitý odběr / dodávka z distribuční sítě právě teď: {_val('grid_power_total_w', 'W')} ({_val('grid_flow_state')})",
        f"- Dnešní odběr ze sítě (denní nákup): {_val('day_grid_buy_kwh', 'kWh')} (celkem nakoupeno: {_val('total_grid_buy_kwh', 'kWh')}, dnes prodáno do sítě: {_val('day_grid_sell_kwh', 'kWh')})",
        f"- Výkon FVE (solární elektrárny) právě teď: {_val('pv_total_power_w', 'W')} (dnes vyrobeno FVE: {_val('day_pv_kwh', 'kWh')}, celkem vyrobeno: {_val('total_pv_kwh', 'kWh')})",
        f"- Baterie FVE: {_val('battery_soc_pct', '%')} ({_val('battery_power_w', 'W')}, {_val('battery_status')})",
        f"- Venkovní teplota: {states.get('sensor.smarthome_venkovni_teplota', 'neznámé')}",
    ]
    return "\n".join(lines)


def parse_assist_stack_ids(dumpsys_output: str) -> list[str]:
    """Extract ActivityManager StackId values for io.homeassistant.companion.android.assist tasks."""
    if not dumpsys_output:
        return []
    stack_ids: list[str] = []
    for line in dumpsys_output.splitlines():
        if ASSIST_TASK_AFFINITY in line:
            match = re.search(r"StackId=(\d+)", line)
            if match:
                sid = match.group(1)
                if sid not in stack_ids:
                    stack_ids.append(sid)
    return stack_ids


def should_trigger_assist_from_log_line(line: str) -> bool:
    """Return True if a logcat line indicates a TCL remote microphone button press."""
    if not line or ASSIST_ACTIVITY in line or "io.homeassistant.companion.android/.assist.AssistActivity" in line:
        return False
    return any(token in line for token in REMOTE_BUTTON_LOG_TRIGGERS)


def should_trigger_assist_from_getevent_line(line: str) -> bool:
    """Return True if a kernel getevent -l line indicates the TCL_RC602S voice button was pressed down."""
    if not line or "EV_KEY" not in line or "DOWN" not in line:
        return False
    return any(key in line for key in REMOTE_GETEVENT_KEY_TRIGGERS)


def build_assist_launch_args() -> list[str]:
    """Build ADB arguments to launch AssistActivity with VOICE_COMMAND and start_listening=true."""
    return [
        "shell",
        "am",
        "start",
        "-a",
        "android.intent.action.VOICE_COMMAND",
        "--ez",
        "start_listening",
        "true",
        "-n",
        ASSIST_ACTIVITY,
    ]


def build_provision_commands() -> list[list[str]]:
    """Build the ordered list of ADB shell commands to provision HA Assist and disable Google Assistant."""
    return [
        # Disable built-in Google Assistant (katniss) so it never steals /dev/hidraw0 or pops over HA Assist
        ["shell", "pm", "disable-user", "--user", "0", KATNISS_PKG],
        ["shell", "am", "force-stop", KATNISS_PKG],
        # Grant audio and overlay permissions to Home Assistant Companion
        ["shell", "pm", "grant", HA_PKG, "android.permission.RECORD_AUDIO"],
        ["shell", "pm", "grant", HA_PKG, "android.permission.POST_NOTIFICATIONS"],
        ["shell", "pm", "grant", HA_PKG, "android.permission.BLUETOOTH_CONNECT"],
        ["shell", "appops", "set", HA_PKG, "SYSTEM_ALERT_WINDOW", "allow"],
        ["shell", "appops", "set", HA_PKG, "RECORD_AUDIO", "allow"],
        ["shell", "dumpsys", "deviceidle", "whitelist", f"+{HA_PKG}"],
        # Set Home Assistant Assist as the default Android Digital Assistant
        ["shell", "settings", "put", "secure", "assistant", ASSIST_ACTIVITY],
        ["shell", "settings", "put", "secure", "voice_interaction_service", VOICE_INTERACTION_SERVICE],
        ["shell", "settings", "put", "secure", "voice_recognition_service", VOICE_INTERACTION_SERVICE],
        # Ensure native TCL on-screen keyboard is active instead of mobile remote bridge bar
        ["shell", "ime", "enable", TCL_IME],
        ["shell", "ime", "set", TCL_IME],
        ["shell", "ime", "disable", REMOTE_BRIDGE_IME],
    ]


def _resolve_tv_target() -> str:
    """Resolve the active ADB target from TCL_TV_ADB_TARGET or auto-detect a connected *:5555 device."""
    env_target = os.environ.get("TCL_TV_ADB_TARGET", "")
    if env_target and "<TV_IP_ADDRESS>" not in env_target:
        return env_target
    if "<TV_IP_ADDRESS>" not in TV_TARGET:
        return TV_TARGET
    res = subprocess.run(["adb", "devices"], capture_output=True, text=True, timeout=5, check=False)
    for line in res.stdout.splitlines():
        if ":5555\tdevice" in line:
            return line.split("\t", 1)[0].strip()
    return TV_TARGET


def run_adb(*args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    target = _resolve_tv_target()
    return subprocess.run(
        ["adb", "-s", target, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def is_tv_authorized() -> bool:
    target = _resolve_tv_target()
    if "<TV_IP_ADDRESS>" not in target:
        subprocess.run(["adb", "connect", target], capture_output=True, text=True, timeout=10, check=False)
    res = subprocess.run(["adb", "devices"], capture_output=True, text=True, timeout=10, check=False)
    for line in res.stdout.splitlines():
        if target in line and "\tdevice" in line:
            return True
    return False


def remove_existing_assist_stacks() -> None:
    """Remove any existing AssistActivity task stack so AssistActivity.onDestroy() cancels old coroutines."""
    res = run_adb("shell", "dumpsys", "activity", "activities", timeout=5)
    stack_ids = parse_assist_stack_ids(res.stdout)
    if stack_ids:
        for sid in stack_ids:
            run_adb("shell", "am", "stack", "remove", sid, timeout=5)
        time.sleep(STACK_REMOVE_SETTLE_SEC)


def inject_text_to_tv_field(text: str, submit: bool = False) -> bool:
    """Dismiss AssistActivity overlay, focus Leanback SearchEditText if on SpeechOrbView, and inject text via ADB."""
    formatted = format_adb_input_text(text)
    if not formatted:
        return False
    print(f"Injecting dictated text into TCL TV input field: {text!r} (formatted={formatted!r}, submit={submit})", flush=True)
    remove_existing_assist_stacks()
    ime_dump = run_adb("shell", "dumpsys", "input_method", timeout=5).stdout
    if "SpeechOrbView" in ime_dump or "TvSpeechOrbView" in ime_dump:
        run_adb("shell", "input", "keyevent", "KEYCODE_DPAD_RIGHT", timeout=5)
        time.sleep(0.35)
    run_adb("shell", "input", "text", formatted, timeout=10)
    if submit:
        time.sleep(0.20)
        run_adb("shell", "input", "keyevent", "KEYCODE_ENTER", timeout=5)
    time.sleep(0.25)
    post_ime_dump = run_adb("shell", "dumpsys", "input_method", timeout=5).stdout
    if "mInputShown=true" in post_ime_dump:
        run_adb("shell", "input", "keyevent", "KEYCODE_BACK", timeout=5)
    return True


def watch_dictation_requests() -> None:
    """Poll DICTATION_REQUEST_FILE every 100ms and inject queued dictation text into the active TV input field."""
    while True:
        try:
            req = consume_dictation_request(DICTATION_REQUEST_FILE)
            if req is not None:
                text = str(req.get("text", "")).strip()
                submit = bool(req.get("submit", False))
                if text:
                    inject_text_to_tv_field(text, submit=submit)
        except Exception as exc:
            print("Dictation request watcher warning:", exc, flush=True)
        time.sleep(0.10)


def trigger_assist_activity(source_desc: str) -> None:
    """Cleanly destroy any stale AssistActivity stack and launch a fresh AssistActivity with immediate mic capture."""
    global _last_trigger_ts
    now = time.time()
    with _trigger_lock:
        if now - _last_trigger_ts <= MIN_DEBOUNCE_SEC:
            return
        _last_trigger_ts = now
    print(f"Detected TCL Remote Mic button press ({source_desc}) -> launching fresh HA AssistActivity...", flush=True)
    remove_existing_assist_stacks()
    run_adb(*build_assist_launch_args(), timeout=10)


def provision_tv() -> bool:
    if not is_tv_authorized():
        return False

    target = _resolve_tv_target()
    pkgs = run_adb("shell", "pm", "list", "packages", HA_PKG, timeout=15)
    if HA_PKG not in pkgs.stdout and APK_PATH.exists():
        print(f"Installing {APK_PATH} onto {target}...", flush=True)
        inst = run_adb("install", "-r", str(APK_PATH), timeout=180)
        print("Install result:", inst.stdout, inst.stderr, flush=True)
        if "Success" not in inst.stdout:
            return False

    for cmd_args in build_provision_commands():
        run_adb(*cmd_args, timeout=15)

    STATE_FILE.write_text(
        json.dumps(
            {
                "status": "provisioned",
                "tv_target": target,
                "package": HA_PKG,
                "katniss_disabled": True,
                "assist_activity": ASSIST_ACTIVITY,
                "voice_interaction_service": VOICE_INTERACTION_SERVICE,
                "default_ime": TCL_IME,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("TCL 65C815 Voice Assistant provisioning completed successfully!", flush=True)
    return True


def watch_remote_getevent() -> None:
    """Monitor Linux input events (/dev/input/event*) via PTY getevent -l for TCL_RC602S KEY_SEARCH DOWN and long-press hold."""
    while True:
        try:
            if not is_tv_authorized():
                time.sleep(5)
                continue
            target = _resolve_tv_target()
            proc = subprocess.Popen(
                ["adb", "-s", target, "shell", "-tt", "getevent", "-l"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
            )
            assert proc.stdout is not None
            key_down_ts: float | None = None
            hold_timer: threading.Timer | None = None

            def _enable_long_press_dictation() -> None:
                print("Detected TCL Remote Mic LONG PRESS (>= 0.65s hold) -> enabling Direct Dictation Mode", flush=True)
                activate_dictation_mode("getevent_hold_timer")

            for line in proc.stdout:
                if should_trigger_assist_from_getevent_line(line):
                    if hold_timer is not None:
                        hold_timer.cancel()
                    clear_dictation_mode()
                    key_down_ts = time.time()
                    hold_timer = threading.Timer(LONG_PRESS_HOLD_SEC, _enable_long_press_dictation)
                    hold_timer.daemon = True
                    hold_timer.start()
                    trigger_assist_activity(f"getevent: {line.strip()}")
                elif is_getevent_search_key_up(line):
                    if hold_timer is not None:
                        hold_timer.cancel()
                        hold_timer = None
                    if key_down_ts is not None:
                        held = time.time() - key_down_ts
                        key_down_ts = None
                        if held >= LONG_PRESS_HOLD_SEC:
                            activate_dictation_mode(f"getevent_hold_{held:.2f}s")
            proc.wait(timeout=5)
        except Exception as exc:
            print("Getevent watcher warning:", exc, flush=True)
            time.sleep(2)


def watch_remote_mic_button() -> None:
    """Intercept TCL remote microphone button presses and SEARCH_LONG_PRESS events via logcat."""
    while True:
        try:
            if not is_tv_authorized():
                time.sleep(5)
                continue
            target = _resolve_tv_target()
            run_adb("logcat", "-c", timeout=10)
            proc = subprocess.Popen(
                ["adb", "-s", target, "logcat", "-v", "brief", "-T", "1"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                if should_activate_dictation_from_log_line(line):
                    print("Detected SEARCH_LONG_PRESS in logcat -> enabling Direct Dictation Mode", flush=True)
                    activate_dictation_mode("logcat_search_long_press")
                elif should_trigger_assist_from_log_line(line):
                    trigger_assist_activity(f"logcat: {line.strip()}")
            proc.wait(timeout=5)
        except Exception as exc:
            print("Logcat watcher warning:", exc, flush=True)
            time.sleep(3)


def main() -> None:
    print("Starting TCL 65C815 Home Assistant Voice Provisioner, Remote Hook & Dictation Worker...", flush=True)
    provision_tv()
    getevent_watcher = threading.Thread(target=watch_remote_getevent, daemon=True)
    getevent_watcher.start()
    logcat_watcher = threading.Thread(target=watch_remote_mic_button, daemon=True)
    logcat_watcher.start()
    dictation_watcher = threading.Thread(target=watch_dictation_requests, daemon=True)
    dictation_watcher.start()
    while True:
        try:
            if not STATE_FILE.exists():
                provision_tv()
            elif is_tv_authorized():
                cur = run_adb("shell", "settings", "get", "secure", "assistant", timeout=10).stdout.strip()
                ime = run_adb("shell", "settings", "get", "secure", "default_input_method", timeout=10).stdout.strip()
                disabled = run_adb("shell", "pm", "list", "packages", "-d", timeout=10).stdout
                if cur != ASSIST_ACTIVITY or ime != TCL_IME or KATNISS_PKG not in disabled:
                    provision_tv()
        except Exception as exc:
            print("Loop warning:", exc, flush=True)
        time.sleep(10)


if __name__ == "__main__":
    main()
