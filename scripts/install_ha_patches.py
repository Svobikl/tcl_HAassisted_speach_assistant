#!/usr/bin/env python3
"""Install TCL Android TV Assist Pipeline & Google Generative AI patches into Home Assistant."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def install_patches(ha_config_dir: Path, container_name: str = "homeassistant") -> None:
    """Copy custom_components and inject assist_pipeline patches into the Home Assistant container."""
    src_cc = REPO_ROOT / "homeassistant" / "custom_components" / "google_generative_ai_conversation"
    dst_cc = ha_config_dir / "custom_components" / "google_generative_ai_conversation"
    dst_cc.parent.mkdir(parents=True, exist_ok=True)
    if dst_cc.exists():
        shutil.rmtree(dst_cc)
    shutil.copytree(src_cc, dst_cc)
    print(f"[OK] Installed custom component -> {dst_cc}")

    ap_dir = REPO_ROOT / "homeassistant" / "assist_pipeline"
    target_in_container = "/usr/src/homeassistant/homeassistant/components/assist_pipeline"
    for fname in ("audio_enhancer.py", "vad.py", "models.py", "default_pipeline.py"):
        src_file = ap_dir / fname
        subprocess.run(
            ["docker", "cp", str(src_file), f"{container_name}:{target_in_container}/{fname}"],
            check=True,
        )
        print(f"[OK] Patched {container_name}:{target_in_container}/{fname}")

    print("[OK] All patches installed. Restart Home Assistant container with: docker restart", container_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Install Home Assistant Assist Pipeline & Gemini patches.")
    parser.add_argument(
        "--ha-config-dir",
        type=Path,
        default=Path("/usr/share/hassio/homeassistant"),
        help="Path to Home Assistant config directory",
    )
    parser.add_argument("--container", default="homeassistant", help="Docker container name (default: homeassistant)")
    args = parser.parse_args()
    install_patches(args.ha_config_dir, args.container)


if __name__ == "__main__":
    main()
