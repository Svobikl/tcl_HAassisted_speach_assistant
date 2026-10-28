"""Unit and negative tests for TCL 65C815 Home Assistant + Gemini voice assistant provisioner and Energy VAD."""

from __future__ import annotations

import array

from tcl_tv_setup_gemini_assistant import (
    ASSIST_ACTIVITY,
    HA_PKG,
    KATNISS_PKG,
    MIN_DEBOUNCE_SEC,
    STACK_REMOVE_SETTLE_SEC,
    TCL_IME,
    VOICE_INTERACTION_SERVICE,
    VOICE_RECORDER_BUFFER_SHORTS,
    EnergySilenceDetector,
    build_assist_launch_args,
    build_immediate_mic_stop_stt_end_payload,
    build_provision_commands,
    compute_pcm16_rms,
    expand_bare_area_command,
    format_smarthome_telemetry_summary,
    normalize_czech_stt_transcript,
    parse_assist_stack_ids,
    patch_voice_recorder_buffer_shorts,
    scan_directory_for_secrets,
    should_trigger_assist_from_getevent_line,
    should_trigger_assist_from_log_line,
    simulate_vad_preroll_and_onset,
    LONG_PRESS_HOLD_SEC,
    activate_dictation_mode,
    clear_dictation_mode,
    consume_dictation_mode,
    consume_dictation_request,
    enqueue_dictation_request,
    format_adb_input_text,
    is_getevent_search_key_up,
    parse_dictation_command,
    peek_dictation_mode,
    should_activate_dictation_from_log_line,
)


def _make_pcm16_chunk(amplitude: int, samples: int = 160) -> bytes:
    """Create a 10ms (160-sample @ 16kHz) 16-bit signed PCM chunk with alternating +/- amplitude."""
    arr = array.array("h", [amplitude if i % 2 == 0 else -amplitude for i in range(samples)])
    return arr.tobytes()


def test_build_provision_commands_disables_katniss_and_sets_ha_assist() -> None:
    """Verify provisioning disables built-in TV Google Assistant and configures HA Assist."""
    cmds = [" ".join(c) for c in build_provision_commands()]
    assert f"shell pm disable-user --user 0 {KATNISS_PKG}" in cmds
    assert f"shell am force-stop {KATNISS_PKG}" in cmds
    assert f"shell pm grant {HA_PKG} android.permission.RECORD_AUDIO" in cmds
    assert f"shell appops set {HA_PKG} SYSTEM_ALERT_WINDOW allow" in cmds
    assert f"shell appops set {HA_PKG} RECORD_AUDIO allow" in cmds
    assert f"shell settings put secure assistant {ASSIST_ACTIVITY}" in cmds
    assert f"shell settings put secure voice_interaction_service {VOICE_INTERACTION_SERVICE}" in cmds
    assert f"shell ime set {TCL_IME}" in cmds


def test_build_assist_launch_args_uses_voice_command_and_start_listening() -> None:
    """Verify AssistActivity launch uses VOICE_COMMAND and start_listening=true for 1st-press mic activation."""
    args = build_assist_launch_args()
    assert "--activity-clear-task" not in args
    assert "android.intent.action.VOICE_COMMAND" in args
    assert "--ez" in args
    assert "start_listening" in args
    assert "true" in args
    assert ASSIST_ACTIVITY in args
    assert 2.0 <= MIN_DEBOUNCE_SEC <= 3.5


def test_parse_assist_stack_ids_happy_and_negative() -> None:
    """Verify StackId extraction for io.homeassistant.companion.android.assist from dumpsys activity output."""
    sample_dumpsys = (
        "  Stack #1: type=home mode=fullscreen\n"
        "    * TaskRecord{c333022 #1269 A=com.google.android.tvlauncher U=0 StackId=1 sz=1}\n"
        "  Stack #98: type=standard mode=fullscreen\n"
        "    * TaskRecord{576a0cf #1360 A=io.homeassistant.companion.android.assist U=0 StackId=98 sz=1}\n"
    )
    assert parse_assist_stack_ids(sample_dumpsys) == ["98"]
    assert parse_assist_stack_ids("") == []
    assert (
        parse_assist_stack_ids(
            "    * TaskRecord{c333022 #1269 A=com.google.android.tvlauncher U=0 StackId=1 sz=1}\n"
        )
        == []
    )


def test_should_trigger_assist_from_log_line_happy_paths() -> None:
    """Verify remote voice button logcat lines trigger AssistActivity launch."""
    assert should_trigger_assist_from_log_line(
        "I/ActivityManager( 2639): START u0 {act=android.speech.action.WEB_SEARCH} from uid 1000"
    )
    assert should_trigger_assist_from_log_line(
        "I/ActivityManager( 2639): START u0 {act=android.speech.action.VOICE_SEARCH_HANDS_FREE} from uid 1000"
    )
    assert should_trigger_assist_from_log_line("I/Katniss: OnConversationTurnStarted")
    assert should_trigger_assist_from_log_line("I/Katniss: Mic button press count: 1")
    assert should_trigger_assist_from_log_line("I/ActivityManager: SearchActivityTrampoline")


def test_should_trigger_assist_from_log_line_negative_cases() -> None:
    """Negative test: empty, unrelated, self-launch, or delayed SEARCH_LONG_PRESS lines must not trigger."""
    assert should_trigger_assist_from_log_line("") is False
    assert should_trigger_assist_from_log_line("D/AudioFlinger( 2310): mixer(0xeb483380) throttle end") is False
    assert (
        should_trigger_assist_from_log_line(
            f"I/ActivityManager( 2639): START u0 {{act=android.intent.action.VOICE_COMMAND cmp={ASSIST_ACTIVITY}}}"
        )
        is False
    )
    assert (
        should_trigger_assist_from_log_line(
            "I/ActivityManager( 2639): START u0 {act=android.intent.action.SEARCH_LONG_PRESS} from uid 1000"
        )
        is False
    )


def test_should_trigger_assist_from_getevent_line_happy_and_negative() -> None:
    """Verify kernel getevent -l line matching for TCL_RC602S KEY_SEARCH DOWN and negative boundaries."""
    assert should_trigger_assist_from_getevent_line(
        "/dev/input/event6: EV_KEY       KEY_SEARCH           DOWN                "
    )
    assert should_trigger_assist_from_getevent_line(
        "/dev/input/event6: EV_KEY       0246                 DOWN                "
    )
    assert (
        should_trigger_assist_from_getevent_line(
            "/dev/input/event6: EV_KEY       KEY_SEARCH           UP                  "
        )
        is False
    )
    assert (
        should_trigger_assist_from_getevent_line(
            "/dev/input/event6: EV_SYN       SYN_REPORT           00000000            "
        )
        is False
    )
    assert (
        should_trigger_assist_from_getevent_line(
            "/dev/input/event6: EV_KEY       KEY_VOLUMEUP         DOWN                "
        )
        is False
    )
    assert should_trigger_assist_from_getevent_line("") is False


def test_compute_pcm16_rms_and_inter_word_pause_preservation() -> None:
    """Verify pre-AGC RMS preserves inter-word pauses ('Rozsviť ... v zádveří') and detects sustained end-of-speech silence."""
    assert compute_pcm16_rms(b"") == 0.0
    assert compute_pcm16_rms(b"\x00") == 0.0
    assert abs(compute_pcm16_rms(_make_pcm16_chunk(200)) - 200.0) < 1e-3
    assert abs(compute_pcm16_rms(_make_pcm16_chunk(2400)) - 2400.0) < 1e-3

    detector = EnergySilenceDetector(
        initial_noise_floor=180.0,
        min_speech_chunks_before_silence_gate=35,
        min_consecutive_low_rms_chunks=22,
    )
    ble_hiss_chunk = _make_pcm16_chunk(160)
    speech_chunk = _make_pcm16_chunk(1800)

    # 1. Pre-speech ambient floor
    for _ in range(10):
        detector.process_chunk(ble_hiss_chunk, raw_vad_prob=0.10)

    # 2. First word ('Rozsviť' - 450ms)
    word1_probs = [detector.process_chunk(speech_chunk, raw_vad_prob=0.92) for _ in range(45)]
    assert all(p > 0.5 for p in word1_probs)

    # 3. Brief inter-word pause (180ms between 'Rozsviť' and 'v zádveří'): must NOT be forced to 0.0
    inter_word_probs = [detector.process_chunk(ble_hiss_chunk, raw_vad_prob=0.85) for _ in range(18)]
    assert all(p > 0.5 for p in inter_word_probs)

    # 4. Second phrase ('v zádveří' - 500ms)
    word2_probs = [detector.process_chunk(speech_chunk, raw_vad_prob=0.95) for _ in range(50)]
    assert all(p > 0.5 for p in word2_probs)

    # 5. Sustained post-speech silence (350ms): must drop speech_probability to 0.0
    post_speech_probs = [detector.process_chunk(ble_hiss_chunk, raw_vad_prob=0.85) for _ in range(35)]
    assert post_speech_probs[-1] == 0.0
    assert sum(1 for p in post_speech_probs[-8:] if p == 0.0) == 8


def test_normalize_czech_stt_transcript_happy_and_negative() -> None:
    """Verify Czech smart-home transcript normalization and noise/silence filtering."""
    assert normalize_czech_stt_transcript("Rozsviť v zádveří.", audio_rms=950.0) == "Rozsviť v zádveří"
    assert normalize_czech_stt_transcript("  Zhasni v obýváku  ", audio_rms=800.0) == "Zhasni v obýváku"
    assert normalize_czech_stt_transcript("<noise>", audio_rms=150.0) == ""
    assert normalize_czech_stt_transcript("<silence>", audio_rms=150.0) == ""
    assert normalize_czech_stt_transcript("", audio_rms=150.0) == ""
    # Low-RMS silence hallucination ("Děkuji." on silent mic) must be rejected as silence
    assert normalize_czech_stt_transcript("Děkuji.", audio_rms=140.0) == ""


def test_patch_voice_recorder_buffer_shorts_happy_and_negative() -> None:
    """Verify VoiceAudioRecorder 10ms (160 shorts) -> 100ms (1600 shorts) DEX patching and checksums."""
    import hashlib
    import struct
    import zlib
    import pytest

    assert VOICE_RECORDER_BUFFER_SHORTS == 1600
    fake_dex = (
        b"dex\n035\x00"
        + b"\x00" * 24
        + b"\xaa" * 64
        + bytes.fromhex("1301803e1302a000")
        + b"\xbb" * 32
        + bytes.fromhex("1308b90128031308a000")
        + b"\xcc" * 32
    )
    patched = patch_voice_recorder_buffer_shorts(fake_dex, target_shorts=1600)
    assert bytes.fromhex("1301803e13024006") in patched
    assert bytes.fromhex("1308b901280313084006") in patched
    assert patched[12:32] == hashlib.sha1(patched[32:]).digest()
    assert struct.unpack_from("<I", patched, 8)[0] == (zlib.adler32(patched[12:]) & 0xFFFFFFFF)

    # Idempotency check (already patched)
    patched_again = patch_voice_recorder_buffer_shorts(patched, target_shorts=1600)
    assert patched_again == patched

    # Negative tests: invalid DEX magic or invalid target_shorts must raise ValueError
    with pytest.raises(ValueError):
        patch_voice_recorder_buffer_shorts(b"not_a_dex", target_shorts=1600)
    with pytest.raises(ValueError):
        patch_voice_recorder_buffer_shorts(fake_dex, target_shorts=0)


def test_energy_silence_detector_does_not_cut_off_quiet_trailing_words_after_loud_onset() -> None:
    """Verify loud initial vowel (5500 RMS) followed by normal trailing words (350 RMS) is never cut off mid-sentence."""
    assert STACK_REMOVE_SETTLE_SEC >= 0.40
    detector = EnergySilenceDetector()
    idle_chunk = _make_pcm16_chunk(60)
    loud_onset_chunk = _make_pcm16_chunk(5500)
    normal_trailing_speech_chunk = _make_pcm16_chunk(350)

    # 1. Idle BLE noise floor (60 RMS for 200ms)
    for _ in range(20):
        detector.process_chunk(idle_chunk, raw_vad_prob=0.15)

    # 2. Loud initial word ('Jaká' / 'Ahoj' at 5500 RMS for 400ms)
    onset_probs = [detector.process_chunk(loud_onset_chunk, raw_vad_prob=0.95) for _ in range(40)]
    assert all(p > 0.5 for p in onset_probs)

    # 3. Inter-word pause (250ms at 60 RMS): must NOT cut off sentence
    pause_probs = [detector.process_chunk(idle_chunk, raw_vad_prob=0.80) for _ in range(25)]
    assert all(p > 0.5 for p in pause_probs)

    # 4. Quieter trailing words ('je denní spotřeba domu' at 350 RMS for 1200ms): must NEVER be forced to 0.0
    trailing_probs = [
        detector.process_chunk(normal_trailing_speech_chunk, raw_vad_prob=0.90) for _ in range(120)
    ]
    assert all(p > 0.5 for p in trailing_probs)

    # 5. Sustained post-speech silence (450ms at 60 RMS): must drop speech_probability to 0.0
    silence_probs = [detector.process_chunk(idle_chunk, raw_vad_prob=0.80) for _ in range(45)]
    assert silence_probs[-1] == 0.0


def test_format_smarthome_telemetry_summary_includes_daily_and_instant_consumption_with_digits() -> None:
    """Verify Smart Home telemetry summary includes both daily consumption (day_load_kwh) and instant power (load_total_w) as digits."""
    deye_attrs = {
        "load_total_w": 530,
        "load_l1_w": 147,
        "load_l2_w": 359,
        "load_l3_w": 24,
        "day_load_kwh": 14.0,
        "total_load_kwh": 29.2,
        "grid_power_total_w": 530,
        "grid_flow_state": "Odběr ze sítě",
        "day_grid_buy_kwh": 14.0,
        "total_grid_buy_kwh": 29.7,
        "day_grid_sell_kwh": 0.2,
        "total_grid_sell_kwh": 2.0,
        "pv_total_power_w": 0,
        "day_pv_kwh": 0.0,
        "total_pv_kwh": 0.0,
        "battery_soc_pct": 0,
        "battery_power_w": 0,
        "battery_status": "Nepřipojena (0 % — připraveno)",
    }
    summary = format_smarthome_telemetry_summary(deye_attrs, {"sensor.smarthome_venkovni_teplota": "8.4 °C"})
    assert "530 W" in summary
    assert "14.0 kWh" in summary
    assert "Denní spotřeba domu dnes" in summary
    assert "Aktuální okamžitý příkon (spotřeba) domu právě teď" in summary

    # Negative boundary: empty/missing attributes must not raise and must fall back cleanly
    empty_summary = format_smarthome_telemetry_summary(None, {})
    assert "neznámé" in empty_summary


def test_energy_silence_detector_rejects_pre_speech_ble_noise_and_recovers_on_consonant_gaps() -> None:
    """Verify idle BLE noise (100 RMS) with high AGC MicroVad prob (0.65) never triggers false speech onset, and active speech with consonant dips stays >= 0.80."""
    detector = EnergySilenceDetector()
    idle_ble_chunk = _make_pcm16_chunk(100)
    active_speech_chunk = _make_pcm16_chunk(650)

    # 1. 1.5s (150 chunks) of pre-speech idle BLE noise where AGC makes raw_vad_prob=0.65: must be forced to 0.0
    pre_speech_probs = [detector.process_chunk(idle_ble_chunk, raw_vad_prob=0.65) for _ in range(150)]
    assert all(p == 0.0 for p in pre_speech_probs)
    assert detector.speech_chunks == 0

    # 2. User starts speaking at 650 RMS (300ms): must immediately register speech
    onset_probs = [detector.process_chunk(active_speech_chunk, raw_vad_prob=0.90) for _ in range(30)]
    assert all(p >= 0.80 for p in onset_probs[-15:])

    # 3. Unvoiced consonant dip where raw_vad_prob drops to 0.20 while vocal energy is still 650 RMS: must stay >= 0.80
    consonant_probs = [detector.process_chunk(active_speech_chunk, raw_vad_prob=0.20) for _ in range(20)]
    assert all(p >= 0.80 for p in consonant_probs)

    # 4. User finishes speaking and returns to 100 RMS idle noise: after 25-chunk grace window, must drop to 0.0
    post_probs = [detector.process_chunk(idle_ble_chunk, raw_vad_prob=0.65) for _ in range(35)]
    assert post_probs[-1] == 0.0
    assert sum(1 for p in post_probs[-8:] if p == 0.0) == 8


def test_build_immediate_mic_stop_stt_end_payload_has_no_text_key() -> None:
    """Verify preliminary stt-end payload turns TV mic icon grey via stopRecording() without adding an empty chat bubble."""
    payload = build_immediate_mic_stop_stt_end_payload()
    assert payload == {"stt_output": {}}
    assert "text" not in payload["stt_output"]


def test_scan_directory_for_secrets_happy_and_negative(tmp_path) -> None:
    """Verify secret scanner passes on clean files and flags Google API keys, GitHub tokens, IPv4 addresses, and personal identifiers."""
    clean_file = tmp_path / "clean.py"
    clean_file.write_text('API_KEY = os.environ.get("GEMINI_API_KEY", "")\n', encoding="utf-8")
    assert scan_directory_for_secrets(tmp_path) == []

    # Negative case: file containing a synthetic Google API key and GitHub PAT must be flagged
    leaked_file = tmp_path / "leaked.env"
    leaked_file.write_text(
        "GOOGLE_KEY=AIza" + "SyD1234567890abcdefghijklmnopqrstuv\n"
        "GH_TOKEN=ghp_" + "1234567890abcdefghijklmnopqrstuvwxyz\n",
        encoding="utf-8",
    )
    findings = scan_directory_for_secrets(tmp_path)
    assert len(findings) >= 2
    assert any("leaked.env" in f for f in findings)


def test_normalize_czech_stt_transcript_repairs_clipped_initial_fricatives() -> None:
    """Verify clipped initial Czech fricatives ('zh-' in 'zhasni', 'roz-' in 'rozsviť') are deterministically repaired without double-z collisions."""
    assert normalize_czech_stt_transcript("asi povídáku", audio_rms=450.0) == "zhasni v obýváku"
    assert normalize_czech_stt_transcript("hlasitě v obýváku", audio_rms=450.0) == "zhasni v obýváku"
    assert normalize_czech_stt_transcript("hasni v obýváku", audio_rms=450.0) == "zhasni v obýváku"
    assert normalize_czech_stt_transcript("zasni v obýváku", audio_rms=450.0) == "zhasni v obýváku"
    assert normalize_czech_stt_transcript("vlastně v obýváku", audio_rms=450.0) == "zhasni v obýváku"
    assert normalize_czech_stt_transcript("asi v kuchyni", audio_rms=450.0) == "zhasni v kuchyni"
    assert normalize_czech_stt_transcript("sviť v obýváku", audio_rms=450.0) == "rozsviť v obýváku"
    assert normalize_czech_stt_transcript("Zhasni v obýváku", audio_rms=450.0) == "Zhasni v obýváku"


def test_expand_bare_area_command_state_aware_toggle_and_negative() -> None:
    """Verify bare room utterances ('v obýváku') expand to Zhasni/Rozsviť based on live light state while explicit commands stay untouched."""
    assert (
        expand_bare_area_command("v obýváku", {"light.ovladani_osvetleni_kuchyne": "on"})
        == "Zhasni světla v obýváku"
    )
    assert (
        expand_bare_area_command("v obýváku", {"light.ovladani_osvetleni_kuchyne": "off"})
        == "Rozsviť světla v obýváku"
    )
    assert (
        expand_bare_area_command("obývák.", {"light.ovladani_osvetleni_kuchyne": "on"})
        == "Zhasni světla v obýváku"
    )
    # Negative cases: explicit verbs or questions must never be rewritten
    assert (
        expand_bare_area_command("zhasni v obýváku", {"light.ovladani_osvetleni_kuchyne": "on"})
        == "zhasni v obýváku"
    )
    assert (
        expand_bare_area_command("rozsviť v obýváku", {"light.ovladani_osvetleni_kuchyne": "on"})
        == "rozsviť v obýváku"
    )
    assert (
        expand_bare_area_command("jaká je teplota v obýváku", {"light.ovladani_osvetleni_kuchyne": "on"})
        == "jaká je teplota v obýváku"
    )
    assert expand_bare_area_command("", {}) == ""


def test_simulate_vad_preroll_and_onset_rejects_10ms_ble_pop_and_preserves_500ms_preroll() -> None:
    """Verify 10ms BLE pop at 400ms is ignored while speech at 800ms triggers VAD and preserves 500ms pre-roll."""
    result = simulate_vad_preroll_and_onset(
        total_pre_speech_chunks=80,
        ble_pop_chunk_index=40,
        speech_chunks_count=60,
        post_silence_chunks_count=50,
    )
    assert result["false_trigger_on_ble_pop"] is False
    assert result["vad_triggered"] is True
    assert result["preroll_chunks_preserved"] == 50
    assert result["ble_pop_discarded_from_preroll"] is True


def test_pipeline_and_stt_files_contain_all_four_voice_fixes() -> None:
    """Verify default_pipeline.py, stt.py, conversation.py, and local_whisper_stt_server.py include all 4 fixes."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    pipeline_text = (repo_root / "homeassistant/assist_pipeline/default_pipeline.py").read_text(encoding="utf-8")
    stt_text = (
        repo_root / "homeassistant/custom_components/google_generative_ai_conversation/stt.py"
    ).read_text(encoding="utf-8")
    conv_text = (
        repo_root / "homeassistant/custom_components/google_generative_ai_conversation/conversation.py"
    ).read_text(encoding="utf-8")
    whisper_text = (repo_root / "scripts/local_whisper_stt_server.py").read_text(encoding="utf-8")

    assert "pre_roll_buffer: deque[bytes] = deque(maxlen=50)" in pipeline_text
    assert "_expand_bare_area_intent_input" in pipeline_text
    assert "mode=local" in stt_text
    assert "_CLIPPED_ZHASNI_RE" in stt_text
    assert "_expand_bare_area_command" in conv_text
    assert "speech_pad_ms" in whisper_text


def test_parse_dictation_command_happy_and_negative() -> None:
    """Verify Czech voice dictation prefix parsing ('Vyplň do pole: Matrix') and negative smart-home guards."""
    assert parse_dictation_command("Vyplň do pole: Matrix") == ("Matrix", False)
    assert parse_dictation_command("vyplň do pole Matrix Reloaded.") == ("Matrix Reloaded", False)
    assert parse_dictation_command("Vyplň pole Matrix") == ("Matrix", False)
    assert parse_dictation_command("Vyplň text Matrix") == ("Matrix", False)
    assert parse_dictation_command("Vyplň Matrix") == ("Matrix", False)
    assert parse_dictation_command("Napiš do pole: Pelíšky") == ("Pelíšky", False)
    assert parse_dictation_command("Napiš text Pelíšky") == ("Pelíšky", False)
    assert parse_dictation_command("Napiš Matrix") == ("Matrix", False)
    assert parse_dictation_command("Zadej do pole Matrix") == ("Matrix", False)
    assert parse_dictation_command("Zadej Matrix") == ("Matrix", False)
    assert parse_dictation_command("Vepiš do pole Matrix") == ("Matrix", False)
    assert parse_dictation_command("Vepiš Matrix") == ("Matrix", False)

    # Search-submit variants (submit=True)
    assert parse_dictation_command("Vyplň a vyhledej Matrix") == ("Matrix", True)
    assert parse_dictation_command("Napiš a vyhledej Pelíšky") == ("Pelíšky", True)
    assert parse_dictation_command("Zadej a vyhledej Matrix") == ("Matrix", True)
    assert parse_dictation_command("Vyhledej v poli Matrix") == ("Matrix", True)

    # Negative cases: smart-home commands, empty payloads, or 'vypni' must never match
    assert parse_dictation_command("") is None
    assert parse_dictation_command("   ") is None
    assert parse_dictation_command("Vyplň do pole:") is None
    assert parse_dictation_command("Napiš do pole") is None
    assert parse_dictation_command("zhasni v obýváku") is None
    assert parse_dictation_command("rozsviť světla v kuchyni") is None
    assert parse_dictation_command("vypni televizi") is None
    assert parse_dictation_command("vypni světlo v obýváku") is None
    assert parse_dictation_command("jaká je denní spotřeba domu") is None
    assert parse_dictation_command("v obýváku") is None


def test_format_adb_input_text_ascii_and_metachar_sanitization() -> None:
    """Verify Czech diacritics are transliterated to ASCII, spaces become %s, and shell metacharacters are stripped."""
    assert format_adb_input_text("Matrix") == "Matrix"
    assert format_adb_input_text("Matrix Reloaded") == "Matrix%sReloaded"
    assert format_adb_input_text("Pelíšky 1999") == "Pelisky%s1999"
    assert format_adb_input_text("Česká televize ČT1") == "Ceska%stelevize%sCT1"
    assert format_adb_input_text("  Matrix; rm -rf /  ") == "Matrix%srm%s-rf%s/"
    assert format_adb_input_text("") == ""
    assert format_adb_input_text("   ") == ""


def test_long_press_detection_and_dictation_queue_lifecycle(tmp_path) -> None:
    """Verify long-press logcat/getevent detection and file-backed dictation mode + request queue lifecycle."""
    assert 0.50 <= LONG_PRESS_HOLD_SEC <= 1.00
    assert should_activate_dictation_from_log_line(
        "I/ActivityManager( 2639): START u0 {act=android.intent.action.SEARCH_LONG_PRESS} from uid 1000"
    )
    assert (
        should_activate_dictation_from_log_line(
            "I/ActivityManager( 2639): START u0 {act=android.speech.action.WEB_SEARCH} from uid 1000"
        )
        is False
    )
    assert should_activate_dictation_from_log_line("") is False

    assert is_getevent_search_key_up("/dev/input/event6: EV_KEY       KEY_SEARCH           UP                  ")
    assert is_getevent_search_key_up("/dev/input/event6: EV_KEY       0246                 UP                  ")
    assert (
        is_getevent_search_key_up("/dev/input/event6: EV_KEY       KEY_SEARCH           DOWN                ")
        is False
    )
    assert (
        is_getevent_search_key_up("/dev/input/event6: EV_KEY       KEY_VOLUMEUP         UP                  ")
        is False
    )

    mode_file = tmp_path / ".tcl_tv_dictation_mode.json"
    req_file = tmp_path / ".tcl_tv_type_request.json"

    assert consume_dictation_mode(mode_file) is False
    activate_dictation_mode("test_long_press", mode_file=mode_file, ttl_sec=15.0)
    assert mode_file.exists()
    assert consume_dictation_mode(mode_file) is True
    # Consuming a second time must return False (one-shot per voice turn)
    assert consume_dictation_mode(mode_file) is False

    # Expired dictation mode flag must return False
    activate_dictation_mode("expired_test", mode_file=mode_file, ttl_sec=-1.0)
    assert consume_dictation_mode(mode_file) is False

    # Clear dictation mode explicitly
    activate_dictation_mode("short_press_cancel", mode_file=mode_file, ttl_sec=15.0)
    clear_dictation_mode(mode_file=mode_file)
    assert consume_dictation_mode(mode_file) is False

    # Dictation typing request queue
    assert consume_dictation_request(req_file) is None
    enqueue_dictation_request("Matrix Reloaded", submit=True, req_file=req_file)
    req = consume_dictation_request(req_file)
    assert req is not None
    assert req["text"] == "Matrix Reloaded"
    assert req["submit"] is True
    assert consume_dictation_request(req_file) is None


def test_peek_dictation_mode_is_nondestructive_for_stt_stage(tmp_path) -> None:
    """Verify peek_dictation_mode lets stt.py inspect active dictation mode without consuming it before conversation.py."""
    mode_file = tmp_path / ".tcl_tv_dictation_mode.json"
    assert peek_dictation_mode(mode_file) is False

    activate_dictation_mode("long_press_mic", mode_file=mode_file, ttl_sec=18.0)
    # Stage 1 (STT) peeks non-destructively
    assert peek_dictation_mode(mode_file) is True
    assert mode_file.exists()
    # Stage 2 (Conversation) consumes and deletes
    assert consume_dictation_mode(mode_file) is True
    assert peek_dictation_mode(mode_file) is False

    # Expired flag must return False on peek
    activate_dictation_mode("expired_mic", mode_file=mode_file, ttl_sec=-1.0)
    assert peek_dictation_mode(mode_file) is False


def test_stt_and_whisper_support_multilingual_dictation_and_resilient_cloud_fallback() -> None:
    """Verify stt.py and local_whisper_stt_server.py implement all 4 dictation & timeout fixes."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    assistant_text = (repo_root / "tcl_tv_setup_gemini_assistant.py").read_text(encoding="utf-8")
    stt_text = (
        repo_root / "homeassistant/custom_components/google_generative_ai_conversation/stt.py"
    ).read_text(encoding="utf-8")
    whisper_text = (repo_root / "scripts/local_whisper_stt_server.py").read_text(encoding="utf-8")

    # Fix 4: YouTube Cobalt VideoSurfaceView focus shift from 'Mluvte...' mic orb before adb input text
    assert "dev.cobalt.media.VideoSurfaceView" in assistant_text

    # Fix 1: Multilingual dictation prompt & non-destructive peek of .tcl_tv_dictation_mode.json
    assert "DICTATION_MULTILINGUAL_STT_PROMPT" in stt_text
    assert "_peek_dictation_mode" in stt_text
    assert "is_dictation: bool = False" in stt_text

    # Fix 2: 4-model Gemini fallback chain, IPv4 preference, >=5.5s/7.5s timeout, and per-model cooldown (no 45s global lockout)
    assert '"models/gemini-2.5-flash"' in stt_text
    assert '"models/gemini-3.6-flash"' in stt_text
    assert "_ipv4_preferred_getaddrinfo" in stt_text
    assert "thinking_budget=0" in stt_text
    assert "DEFAULT_GEMINI_STT_TIMEOUT_SEC = 5.5" in stt_text
    assert "DICTATION_GEMINI_STT_TIMEOUT_SEC = 7.5" in stt_text
    assert "_model_cooldown_until" in stt_text
    assert "+ 45.0" not in stt_text

    # Fix 3: ThreadingHTTPServer, temperature=0.0, without_timestamps=True, dictation=1, BrokenPipeError guard
    assert "ThreadingHTTPServer" in whisper_text
    assert "temperature=0.0" in whisper_text
    assert "without_timestamps=True" in whisper_text
    assert "dictation" in whisper_text
    assert "BrokenPipeError" in whisper_text


def test_normalize_czech_stt_transcript_preserves_english_in_dictation_mode() -> None:
    """Verify normalize_czech_stt_transcript skips Czech smart-home phonetic rewrites when is_dictation=True."""
    assert (
        normalize_czech_stt_transcript("LLM inference on MacBook.", audio_rms=600.0, is_dictation=True)
        == "LLM inference on MacBook"
    )
    assert (
        normalize_czech_stt_transcript("Matt Armstrong", audio_rms=600.0, is_dictation=True)
        == "Matt Armstrong"
    )
    # In dictation mode, even a phrase matching _CLIPPED_ZHASNI_RE must not be rewritten into 'zhasni ...'
    assert (
        normalize_czech_stt_transcript("asi v obýváku", audio_rms=600.0, is_dictation=True)
        == "asi v obýváku"
    )
    # Silence and low-RMS hallucinations are still filtered in dictation mode
    assert normalize_czech_stt_transcript("<noise>", audio_rms=150.0, is_dictation=True) == ""
    assert normalize_czech_stt_transcript("Děkuji.", audio_rms=120.0, is_dictation=True) == ""


def test_repository_contains_zero_secrets() -> None:
    """Ensure the entire repository tree contains zero API keys, tokens, IPv4 addresses, or personal identifiers."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    assert scan_directory_for_secrets(repo_root) == []
