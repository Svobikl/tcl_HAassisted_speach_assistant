#!/usr/bin/env python3
"""Patch Home Assistant Companion APK for Android TV Bluetooth Remote Microphones (100ms chunks & AudioSource=6)."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tcl_tv_setup_gemini_assistant import patch_voice_recorder_buffer_shorts


def patch_apk(input_apk: Path, output_apk: Path, buffer_shorts: int = 1600) -> None:
    """Patch classes7.dex inside the Home Assistant Companion APK for 1.0x real-time TV streaming."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        with zipfile.ZipFile(input_apk, "r") as zin:
            zin.extractall(tmp)
        dex7 = tmp / "classes7.dex"
        if not dex7.exists():
            raise FileNotFoundError("classes7.dex not found in APK")
        dex7.write_bytes(patch_voice_recorder_buffer_shorts(dex7.read_bytes(), target_shorts=buffer_shorts))
        with zipfile.ZipFile(output_apk, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for file_path in sorted(tmp.rglob("*")):
                if file_path.is_file():
                    arcname = file_path.relative_to(tmp).as_posix()
                    compress = zipfile.ZIP_STORED if arcname.endswith(".so") or arcname == "resources.arsc" else zipfile.ZIP_DEFLATED
                    zout.write(file_path, arcname, compress_type=compress)
    print(f"Patched APK written to {output_apk} (re-align with zipalign -p 4 and sign with apksigner).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch HA Companion APK for Android TV BLE voice remotes.")
    parser.add_argument("--input", required=True, type=Path, help="Path to original HA Companion APK")
    parser.add_argument("--output", required=True, type=Path, help="Path to output patched APK")
    parser.add_argument("--buffer-shorts", type=int, default=1600, help="PCM16 shorts per chunk (default: 1600 = 100ms)")
    args = parser.parse_args()
    patch_apk(args.input, args.output, buffer_shorts=args.buffer_shorts)


if __name__ == "__main__":
    main()
