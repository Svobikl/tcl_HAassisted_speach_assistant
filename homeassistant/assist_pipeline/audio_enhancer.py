"""Audio enhancer for Assist Pipeline with Pre-AGC RMS Speech Onset & Silence Detection."""

from abc import ABC, abstractmethod
import array
import math
import sys

from pymicro_vad import MicroVad
from pyspeex_noise import AudioProcessor

from .const import SAMPLE_CHANNELS, SAMPLE_RATE, SAMPLE_WIDTH


def _compute_pcm16_rms(audio: bytes) -> float:
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


class EnhancedAudioChunk:
    """Audio chunk with enhanced audio and speech probability."""

    __slots__ = ("audio", "speech_probability", "timestamp_ms")

    def __init__(
        self, audio: bytes, timestamp_ms: int, speech_probability: float | None
    ) -> None:
        """Initialize enhanced audio chunk."""
        self.audio = audio
        self.timestamp_ms = timestamp_ms
        self.speech_probability = speech_probability


class AudioEnhancer(ABC):
    """Base class for audio enhancement and voice activity detection."""

    def __init__(
        self, auto_gain: int, noise_suppression: int, is_vad_enabled: bool
    ) -> None:
        """Initialize audio enhancer."""
        self.auto_gain = auto_gain
        self.noise_suppression = noise_suppression
        self.is_vad_enabled = is_vad_enabled

    @abstractmethod
    def enhance_chunk(self, audio: bytes, timestamp_ms: int) -> EnhancedAudioChunk:
        """Enhance chunk of 16Khz 16-bit mono PCM audio and return chunk with enhancements applied."""


class MicroVadSpeexEnhancer(AudioEnhancer):
    """Audio enhancer that uses pymicro_vad, pyspeex_noise, and Pre-AGC RMS Onset + Silence Detection."""

    def __init__(
        self, auto_gain: int, noise_suppression: int, is_vad_enabled: bool
    ) -> None:
        """Initialize audio enhancer."""
        super().__init__(auto_gain, noise_suppression, is_vad_enabled)
        self.audio_processor: AudioProcessor | None = None

        if (self.noise_suppression > 0) or (self.auto_gain > 0):
            noise_suppression_db = self.noise_suppression * -10
            auto_gain_dbfs = 2 ** (self.auto_gain // 2)
            self.audio_processor = AudioProcessor(
                auto_gain_dbfs, noise_suppression_db
            )

        self.vad: MicroVad | None = None
        if self.is_vad_enabled:
            self.vad = MicroVad()

        self.min_raw_rms: float = 120.0
        self.ema_raw_rms: float = 120.0
        self.peak_raw_rms: float = 0.0
        self.speech_chunks: int = 0
        self.consecutive_onset_chunks: int = 0
        self.consecutive_low_rms_chunks: int = 0
        self.chunks_seen: int = 0

    def enhance_chunk(self, audio: bytes, timestamp_ms: int) -> EnhancedAudioChunk:
        """Enhance 10ms chunk of 16Khz 16-bit mono PCM audio and return chunk with enhancements applied."""
        raw_rms = _compute_pcm16_rms(audio)
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

        if self.audio_processor is not None:
            audio = self.audio_processor.Process10ms(audio).audio

        raw_vad_prob: float | None = None
        if self.vad is not None:
            raw_vad_prob = self.vad.Process10ms(audio)

        speech_probability = raw_vad_prob
        if speech_probability is not None:
            if self.speech_chunks == 0:
                if max(raw_rms, self.ema_raw_rms) < onset_threshold or speech_probability <= 0.35:
                    self.consecutive_onset_chunks = max(0, self.consecutive_onset_chunks - 1)
                    speech_probability = 0.0
                else:
                    self.consecutive_onset_chunks += 1
                    # Require at least 50ms (5 chunks) of sustained speech energy before locking into in_speech
                    # so a 10ms BLE mic startup pop never triggers stt-vad-start prematurely.
                    if self.consecutive_onset_chunks >= 5:
                        self.speech_chunks = self.consecutive_onset_chunks
                        self.consecutive_low_rms_chunks = 0
                        self.peak_raw_rms = max(capped_ema, min(raw_rms, 3500.0))
                    speech_probability = max(speech_probability, 0.85)
            else:
                if capped_ema > self.peak_raw_rms:
                    self.peak_raw_rms = capped_ema
                silence_threshold = min(
                    245.0,
                    max(
                        150.0,
                        self.min_raw_rms * 1.38,
                        self.peak_raw_rms * 0.060,
                    ),
                )
                if self.ema_raw_rms >= silence_threshold:
                    self.consecutive_low_rms_chunks = 0
                    self.speech_chunks += 1
                    speech_probability = max(speech_probability, 0.85)
                elif self.speech_chunks < 35:
                    self.speech_chunks += 1
                    speech_probability = max(speech_probability, 0.80)
                else:
                    self.consecutive_low_rms_chunks += 1
                    if self.consecutive_low_rms_chunks >= 35:
                        speech_probability = 0.0
                    else:
                        speech_probability = max(speech_probability, 0.80)

        return EnhancedAudioChunk(
            audio=audio,
            timestamp_ms=timestamp_ms,
            speech_probability=speech_probability,
        )
