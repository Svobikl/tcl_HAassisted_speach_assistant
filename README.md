# TCL Android TV — Home Assistant & Gemini / Faster-Whisper Voice Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Android%20TV%20%7C%20Raspberry%20Pi%205%20%7C%20Home%20Assistant-green.svg)]()

Complete, production-tested integration that replaces the stock Google Assistant on **TCL Android TV** (tested on **TCL 65C815** with the **TCL RC602S** Bluetooth BLE voice remote) with **Home Assistant Assist** powered by **Google Gemini (`gemini-3.5-flash-lite` / `gemini-3.8-live`)** and a warm **Local `faster-whisper` Fallback Daemon**.

*(Český návod a kompletní popis architektury najdete v druhé polovině dokumentu níže.)*

---

## Key Problems Solved for the Community

Running the official Home Assistant Companion app (`io.homeassistant.companion.android`) as a push-to-talk voice assistant on Android TV with a Bluetooth BLE remote microphone suffers from **six major hardware and software bottlenecks** out of the box. This repository solves all six:

1. **Built-in Google Assistant (`com.google.android.katniss`) Hijacking the BLE Remote Mic:**
   - On TCL Android TVs, pressing the remote microphone button triggers `katniss`, which locks `/dev/hidraw0` (`libiflyvoice`).
   - **Fix:** `tcl_tv_setup_gemini_assistant.py` disables `com.google.android.katniss`, binds Home Assistant Assist as the default system digital assistant, and watches both kernel `getevent -l` (`KEY_SEARCH` / `KEY_VOICE_ASSIST`) and `logcat` triggers with a `2.5s` debounce.

2. **5.12x Slower-than-Real-Time Audio Streaming on Android TV (`160` -> `1600` Shorts DEX Patch):**
   - By default, `VoiceAudioRecorder` in `ha-companion` reads `160` PCM16 shorts (`10ms` = `100 chunks/sec`). However, `AssistViewModelBase$setupRecorder$1` performs a synchronous SQLite query (`serverManager.webSocketRepository`) + WebSocket dispatch per chunk (~`51ms`/chunk on Android TV SoCs), causing a `5.12x` backlog where a 3-second voice command takes 15+ seconds to transmit.
   - Furthermore, Android TV ROMs often omit the `android.hardware.microphone` feature flag and require `AudioSource.VOICE_RECOGNITION (6)` to route `AUDIO_DEVICE_IN_BLUETOOTH_BLE`.
   - **Fix:** `scripts/patch_ha_companion_apk.py` patches `classes7.dex` to use `1600` shorts (`100ms` = `3200 bytes` = `10 chunks/sec`), achieving true **1.0x real-time streaming** while Home Assistant's `chunk_samples` transparently slices each 100ms packet into ten 10ms VAD frames.

3. **TV Microphone Icon Staying Blue ("Still Listening") After You Stop Speaking:**
   - Bytecode analysis of `ha-companion` (`classes7.dex`, `AssistViewModelBase.handleSttEnd` at `0x28181c`) revealed that the Android app **completely ignores `stt-vad-end`** (`WebSocketCoreImpl: Unknown event type ignoring`) and only calls `stopRecording()` when it receives `stt-end`.
   - **Fix:** `homeassistant/assist_pipeline/default_pipeline.py` emits a preliminary `{"type": "stt-end", "data": {"stt_output": {}}}` at the exact moment `stt-vad-end` fires. Because `"text"` is absent from `stt_output`, `handleSttEnd` immediately invokes `stopRecording()` (turning the TV microphone icon from blue to grey in **`0 ms`**) without adding an empty chat bubble. When transcription finishes ~`1.1s` later, the final `stt-end` renders the user's transcribed text bubble.

4. **Pre-Speech BLE Noise False Triggers, Initial Fricative Clipping & Mid-Sentence Cutoffs (`500 ms` Pre-Roll + `EnergySilenceDetector`):**
   - `pyspeex_noise` automatic gain control (`auto_gain=1`) amplifies idle Bluetooth remote noise (`~100 RMS`) up to `~800 RMS`, tricking `webrtc-noise-gain` MicroVad into firing `stt-vad-start` before the user starts speaking, while soft Czech initial consonant clusters (`zh-` in `"zhasni v obýváku"`) are attenuated during the first `~200 ms` after button press.
   - **Fix:**
     - `homeassistant/assist_pipeline/default_pipeline.py` maintains a **`500 ms` (`50 x 10 ms`) pre-roll ring buffer** (`pre_roll_buffer: deque[bytes] = deque(maxlen=50)`) prior to `stt-vad-start`. All earlier pre-speech BLE startup pops and silence are discarded, while the `500 ms` leading up to speech onset is flushed into STT so initial fricatives (`"zhasni"`) are never clipped.
     - `homeassistant/assist_pipeline/audio_enhancer.py` requires `5` sustained onset chunks (`50 ms` with 1-chunk gap tolerance) above `max(165.0, min(320.0, min_raw_rms * 1.45))` so 10ms BLE mic startup pops never trigger `stt-vad-start` prematurely, and holds `350 ms` across inter-word pauses (`"zhasni ... v obýváku"`).

5. **Czech Phonetic Post-Correction & State-Aware Bare-Room Ellipsis Expansion (`"v obýváku"`):**
   - If the user presses the remote mic button late (after already saying `"zhasni"`) so only `"v obýváku"` is captured on the microphone, or if BLE Speex noise suppression clips `"zhasni"` into `"hasni"` / `"zasni"` / `"asi povídáku"`, standard assistants fail to execute the command.
   - **Fix:**
     - `stt.py`, `local_whisper_stt_server.py`, and `tcl_tv_setup_gemini_assistant.py` apply deterministic regex post-correction (`_CLIPPED_ZHASNI_RE`, `_CLIPPED_ROZSVIT_RE`) for clipped Czech fricatives before intent matching.
     - `default_pipeline.py` (`_expand_bare_area_intent_input`) and `conversation.py` (`_expand_bare_area_command`) inspect the live state of the room's lights in Home Assistant (`hass.states`) whenever a bare room name (`"v obýváku"`, `"v kuchyni"`, `"v zádveří"`, `"v ložnici"`, etc.) is received, automatically expanding it into `"Zhasni světla v obýváku"` (if lights are `on`) or `"Rozsviť světla v obýváku"` (if lights are `off`).

6. **Low-Latency Gemini STT with Zero-Delay `?mode=local` `faster-whisper` Fallback:**
   - Primary STT uses `models/gemini-3.5-flash-lite` (`~1.05s – 1.30s` response time).
   - If the cloud API times out or hits rate limits (`429`), `homeassistant/custom_components/google_generative_ai_conversation/stt.py` calls `http://localhost:8765/transcribe?lang=cs&mode=local` (`scripts/local_whisper_stt_server.py`), which bypasses any redundant cloud retry and runs `faster-whisper` (`base` `int8`) with non-clipping `vad_parameters` (`threshold=0.20`, `speech_pad_ms=800`).

---

## Architecture Flow

```mermaid
flowchart LR
    A["TCL RC602S Remote
(KEY_SEARCH + BLE Mic)"] -->|ADB getevent / logcat| B["tcl-tv-assistant.service
(Stack Cleanup + Launch)"]
    B -->|VOICE_COMMAND
start_listening=true| C["Patched HA Companion APK
(1600 shorts = 100ms chunks)"]
    C -->|1.0x Real-Time PCM16| D["HA Assist Pipeline
(500ms Pre-Roll + Pre-AGC VAD)"]
    D -->|Immediate stt-end {}| C
    D -->|Audio Stream| E["Gemini 3.5 Flash-Lite STT
(~1.1s + Czech Post-Correction)"]
    E -->|Fallback ?mode=local| F["Warm Local Faster-Whisper
(localhost:8765)"]
    E --> G["State-Aware Ellipsis Expansion +
HA Intents / Gemini 3.8 Live"]
    F --> G
    G -->|TTS + UI Reply| C
```

---

## Repository Structure

```text
├── tcl_tv_setup_gemini_assistant.py      # Main ADB provisioner, remote button watcher & EnergySilenceDetector
├── scripts/
│   ├── install_ha_patches.py             # Automated installer for HA assist_pipeline & custom_components
│   ├── local_whisper_stt_server.py       # Persistent warm faster-whisper HTTP fallback server (localhost:8765)
│   └── patch_ha_companion_apk.py         # Binary DEX patcher for HA Companion APK (160 -> 1600 shorts)
├── homeassistant/
│   ├── assist_pipeline/                  # Patched HA Core assist_pipeline modules (audio_enhancer, vad, models, default_pipeline)
│   └── custom_components/
│       └── google_generative_ai_conversation/ # Enhanced Gemini STT + Conversation integration with local Whisper fallback
├── systemd/
│   ├── tcl-tv-assistant.service          # Systemd unit for the TCL TV ADB remote hook & provisioner
│   └── local-whisper-stt.service         # Systemd unit for the warm local faster-whisper daemon
├── tests/
│   └── test_tcl_tv_voice_assistant.py    # Automated pytest suite (19 unit, boundary, DEX, pre-roll & secret-scan tests)
├── .env.example                          # Example configuration file (no secrets or IPs)
├── requirements.txt                      # Python dependencies
└── LICENSE                               # MIT License
```

---

## Quick Start & Installation Guide (English)

### 1. Prerequisites
- **Android TV** (e.g., TCL 65C815) with **Network ADB Debugging** enabled (`Settings -> Device Preferences -> Developer Options -> Network debugging`).
- **Home Assistant** (running in Docker / Supervised on Raspberry Pi 5 or Linux host) with `adb` (`android-tools-adb`) installed on the host.

### 2. Patch the Home Assistant Companion APK
Download the official `io.homeassistant.companion.android` APK, patch `classes7.dex` for 100ms (`1600`-short) BLE audio streaming, align, and sign:
```bash
python3 scripts/patch_ha_companion_apk.py \
  --input ha-companion-original.apk \
  --output ha-companion-tv-patched.apk

zipalign -f -p 4 ha-companion-tv-patched.apk ha-companion-tv.apk
apksigner sign --ks your-debug.keystore --ks-pass pass:android ha-companion-tv.apk
cp ha-companion-tv.apk /usr/share/hassio/homeassistant/www/ha-companion-tv.apk
```

### 3. Install Home Assistant Pipeline & Gemini Patches
```bash
python3 scripts/install_ha_patches.py \
  --ha-config-dir /usr/share/hassio/homeassistant \
  --container homeassistant

docker restart homeassistant
```

### 4. Install & Enable Systemd Services
```bash
sudo cp tcl_tv_setup_gemini_assistant.py /usr/share/hassio/homeassistant/
sudo cp scripts/local_whisper_stt_server.py /usr/local/bin/
sudo chmod +x /usr/local/bin/local_whisper_stt_server.py

sudo cp systemd/tcl-tv-assistant.service /etc/systemd/system/
sudo cp systemd/local-whisper-stt.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now local-whisper-stt.service
sudo systemctl enable --now tcl-tv-assistant.service
```

### 5. Run Automated Tests
```bash
pytest -v
```

---

## Český návod k instalaci a popis řešení (CZ)

### Co tento projekt řeší?
Tento repozitář obsahuje kompletní sadu skriptů a úprav pro plnohodnotného českého hlasového asistenta na televizích **TCL Android TV** (ověřeno na **TCL 65C815** s Bluetooth dálkovým ovladačem **TCL RC602S**) napojeného na **Home Assistant Assist**, **Google Gemini (`gemini-3.5-flash-lite` / `gemini-3.8-live`)** a lokální záložní **Faster-Whisper STT**.

### Hlavní technické vychytávky
1. **Okamžité zešednutí mikrofonu po domluvení (`0 ms`):**
   - Aplikace Home Assistant Companion na Android TV ignoruje událost `stt-vad-end` a vypíná nahrávání (`stopRecording()`) teprve po přijetí `stt-end`. Upravený `default_pipeline.py` proto v okamžiku detekce ticha (`stt-vad-end`) okamžitě odešle předběžný `stt-end` s prázdným `{"stt_output": {}}`, čímž modrá ikona mikrofonu ihned zešedne bez vytvoření prázdné bubliny.
2. **`500 ms` Pre-Roll kruhový buffer + Pre-AGC detekce energie hlasu (`EnergySilenceDetector`):**
   - Udržuje `500 ms` (`50 x 10 ms`) před-buffer audia před spuštěním `stt-vad-start`, takže tiché české souhlásky na začátku věty (`"zh-"` ve slově `"zhasni"`) se nikdy neoříznou, zatímco úvodní lupnutí Bluetooth mikrofonu (`~10 ms`) je odfiltrováno (`consecutive_onset_chunks >= 5`).
3. **Fonetická post-korekce a kontextové doplnění samotné místnosti (`"v obýváku"`):**
   - Pokud uživatel stiskne tlačítko mikrofonu později a do mikrofonu se zachytí jen `"v obýváku"`, pipeline se v reálném čase podívá na stav světel v dané místnosti (`hass.states`) a povel automaticky rozvine na `"Zhasni světla v obýváku"` (pokud svítí) nebo `"Rozsviť světla v obýváku"` (pokud jsou zhasnutá).
4. **1.0x Real-Time přenos hlasu z TV (`1600` shorts = `100ms` bloky):**
   - Úprava bufferu `VoiceAudioRecorder` v `classes7.dex` ze `160` shorts (`10ms`) na `1600` shorts (`100ms`) odstraňuje zpoždění SQLite+WebSocket fronty na Android TV.
5. **Rychlé Gemini STT (`~1.1 s`) + přímý `?mode=local` Faster-Whisper fallback:**
   - Primární přepis řeči zajišťuje `models/gemini-3.5-flash-lite` (`~1.1 s`), při výpadku internetu nebo vyčerpání kvóty (`429`) okamžitě přebírá přepis lokální démon `local-whisper-stt.service` (`localhost:8765/transcribe?lang=cs&mode=local`) bez duplicitního čekání na cloud.

### Postup instalace v kostce
1. Nastav adresu své TV v `/etc/systemd/system/tcl-tv-assistant.service` (`TCL_TV_ADB_TARGET=<TV_IP_ADDRESS>:5555`).
2. Nasaď upravené moduly do Home Assistanta pomocí `python3 scripts/install_ha_patches.py` a restartuj kontejner `homeassistant`.
3. Povol a spusť systémové služby `local-whisper-stt.service` a `tcl-tv-assistant.service`.
4. V Home Assistantovi (`Nastavení -> Integrace -> Google Generative AI Conversation`) zadej svůj vlastní Google Gemini API klíč.

---

## License

Released under the [MIT License](LICENSE).
