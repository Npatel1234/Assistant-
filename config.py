"""Settings from .env (keys case-insensitive) plus defaults. Store UTC, show TZ."""
import os
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).parent

DEFAULTS = {
    "GEMINI_API_KEY": "",
    "MODELS": "gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-3.5-flash,gemini-2.5-flash",
    "EMBED_MODEL": "gemini-embedding-2",
    "SEARCH_MODELS": "gemini-2.5-flash",
    "RPM_LIMIT": "15",
    "APP_PASSWORD": "",
    "SESSION_SECRET": "",
    "PORT": "8765",
    "TIMEZONE": "Asia/Kolkata",
    "CLAUDE_TIMEOUT": "3600",
    "CLAUDE_WAIT": "180",
    "REPO_TEST_TIMEOUT": "300",
    "CLAUDE_IDLE_MIN": "30",
    "CLAUDE_MAX_SESSIONS": "3",
    "TERMINAL_IDLE_MIN": "30",
    "SYNC_MAX_CALLS": "30",
    "DB_PATH": str(ROOT / "assistant.db"),
    "CLAUDE_HISTORY_DIR": str(Path.home() / ".claude" / "projects"),
}


def load_env(path: Path = ROOT / ".env") -> dict[str, str]:
    """KEY=VALUE lines, keys upper-cased (so `gemini_api_key` works). Real env vars override."""
    env = dict(DEFAULTS)
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            # ponytail: " #" starts an inline comment, so values can't contain " #"
            env[key.strip().upper()] = value.split(" #")[0].strip().strip("'\"")
    for key, value in os.environ.items():
        if key.upper() in DEFAULTS:
            env[key.upper()] = value
    return env


def _list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


_env = load_env()
GEMINI_API_KEY = _env["GEMINI_API_KEY"]
MODELS = _list(_env["MODELS"])
EMBED_MODEL = _env["EMBED_MODEL"]
SEARCH_MODELS = _list(_env["SEARCH_MODELS"])
RPM_LIMIT = int(_env["RPM_LIMIT"])
APP_PASSWORD = _env["APP_PASSWORD"]
SESSION_SECRET = _env["SESSION_SECRET"]
PORT = int(_env["PORT"])
TZ = ZoneInfo(_env["TIMEZONE"])  # machine clock is UTC; all user-facing time uses this
CLAUDE_TIMEOUT = int(_env["CLAUDE_TIMEOUT"])
CLAUDE_WAIT = int(_env["CLAUDE_WAIT"])
REPO_TEST_TIMEOUT = int(_env["REPO_TEST_TIMEOUT"])
CLAUDE_IDLE_MIN = int(_env["CLAUDE_IDLE_MIN"])
CLAUDE_MAX_SESSIONS = int(_env["CLAUDE_MAX_SESSIONS"])
TERMINAL_IDLE_MIN = int(_env["TERMINAL_IDLE_MIN"])
SYNC_MAX_CALLS = int(_env["SYNC_MAX_CALLS"])
DB_PATH = _env["DB_PATH"]
CLAUDE_HISTORY_DIR = Path(_env["CLAUDE_HISTORY_DIR"])
