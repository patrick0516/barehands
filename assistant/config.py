# -*- coding: utf-8 -*-
"""Trimmed, cross-platform config for Jarvis's ported voice assistant.

Ported from sadie-voice/config.py. Differences from the original, on
purpose:
  - Paths use pathlib and expanduser() properly (the original had a raw
    `~\\.sadie-voice` string that only worked on Windows).
  - OPENCODE is resolved via PATH lookup instead of a hardcoded Windows
    npm path.
  - MIC_INDEX/OUT_DEVICE default to None (system default device) instead
    of a Windows device-name keyword ("Microsoft").
  - No Gemini OAuth-token-refresh machinery: auth is a plain
    GEMINI_API_KEY env var. sadie-voice's token-file flow is more capable
    but is extra infra beyond "get the conversation working" for this
    first pass; revisit if a longer-lived credential is needed later.
"""
import os
import shutil
from pathlib import Path


def _load_dotenv(path):
    """Minimal KEY=VALUE loader, no dependency on python-dotenv. Only
    fills in vars that aren't already set in the real environment, so a
    real env var always wins over the file."""
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
    except FileNotFoundError:
        pass
    except Exception:
        pass


_load_dotenv(Path(__file__).resolve().parent.parent / ".env")

ASSISTANT_NAME = "Sadie"  # kept as Sadie on purpose -- see DECISIONS.md

# vosk's grammar mode can ONLY recognize words that exist in the model's
# own vocabulary -- it can't just be told a new word and hear it. Checked
# directly against the downloaded vosk-model-small-cn-0.22
# (KaldiRecognizer logs "Ignoring word missing in vocabulary" for any
# grammar word it can't back): "sadie", "賽迪", "薩迪", "塞迪", "莎蒂" are
# ALL out of vocabulary in this small model -- the wake word would
# silently never fire with those. "小迪"/"迪迪" (natural short nicknames
# for Sadie) ARE in vocabulary and confirmed working. If you want to wake
# it by saying "Sadie" itself, swap in the full-size vosk-model-cn-0.22
# (~1.3GB vs. this model's ~65MB) from https://alphacephei.com/vosk/models
# and re-check these aliases against it the same way.
WAKE_ALIASES = ("小迪", "迪迪")
WAKE_HINT = "小迪 / 迪迪"

APP_DIR = Path(__file__).resolve().parent
BAREHANDS_DIR = APP_DIR.parent
STATE_DIR = BAREHANDS_DIR / "state"
WAKE_MODEL = APP_DIR / "wakeword-model"  # not tracked in git; download separately


def _find_opencode():
    p = shutil.which("opencode")
    if p:
        return p
    for candidate in (
        Path.home() / ".opencode" / "bin" / "opencode",
        Path.home() / "AppData" / "Roaming" / "npm" / "node_modules"
        / "opencode-ai" / "bin" / "opencode.exe",
    ):
        if candidate.exists():
            return str(candidate)
    return None


OPENCODE = _find_opencode()
WORKDIR = os.environ.get("SADIE_WORKDIR", str(Path.home() / ".jarvis-assistant"))

LIVE_MODEL = os.environ.get("GEMINI_LIVE_MODEL", "gemini-3.1-flash-live-preview")
VOICE = os.environ.get("SADIE_VOICE", "Kore")
MIC_INDEX = os.environ.get("SADIE_MIC")          # None = system default input
OUT_DEVICE = os.environ.get("SADIE_OUT")          # None = system default output
PLAY_GAIN = float(os.environ.get("SADIE_VOLUME", "1.5"))
RECONNECT_MAX = int(os.environ.get("SADIE_RECONNECT", "3"))


def gemini_key():
    """Plain env-var credential. See module docstring for why this is
    simpler than sadie-voice's OAuth-refresh flow."""
    return os.environ.get("GEMINI_API_KEY", "")
