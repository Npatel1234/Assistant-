# Gemini Personal Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A LAN web app where Gemini (free tier) acts as a proactive personal assistant with short/long-term memory, tracks projects/ideas/reminders, syncs from Claude Code history, and delegates code questions to Claude Code (Pro) in project folders.

**Architecture:** One FastAPI process (`uv run main.py`) serves a vanilla-JS single page, a JSON API, an SSE chat endpoint, and an asyncio background scheduler. SQLite (WAL) holds everything; embeddings are float32 blobs searched by numpy cosine. `llm.LLM` wraps `google-genai` with an ordered model fallback chain; `claude_runner.ClaudeRunner` shells out to `claude -p`.

**Tech Stack:** Python ≥3.11, uv, FastAPI, uvicorn, google-genai 2.x, httpx, itsdangerous, numpy, stdlib sqlite3, pytest.

**Spec:** `docs/superpowers/specs/2026-09-29-gemini-assistant-design.md` (§16 "Revisions" overrides earlier sections).

## Global Constraints

- Python ≥3.11; run with `uv run main.py`; tests with `uv run pytest -q`.
- Dependencies limited to: fastapi, uvicorn, google-genai, httpx, itsdangerous, numpy (+ pytest dev). No others.
- Store every timestamp as UTC ISO via `db.now()` / `db.to_utc()`. Interpret and display user time in `config.TZ` (default `Asia/Kolkata`). Never call `datetime.now()` without a tz; machine timezone is UTC and must not be changed.
- Default `MODELS=gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-3.5-flash,gemini-2.5-flash`, `EMBED_MODEL=gemini-embedding-2`, `SEARCH_MODELS=gemini-2.5-flash`, `RPM_LIMIT=15`, `PORT=8765`, `CLAUDE_TIMEOUT=600`, `SYNC_MAX_CALLS=30`.
- `.env` keys are case-insensitive (user has `gemini_api_key=`). Never print or log the API key. Never commit `.env`.
- google-genai client built with `http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1))`; fallback is ours.
- Claude Code: `claude -p --output-format json --permission-mode bypassPermissions [--resume ID]`, prompt on **stdin**, `cwd` = registered project path only, one run at a time, `CLAUDECODE` env var removed.
- Server binds `0.0.0.0:8765`. Every route except `/login` requires the signed session cookie.
- All tests live in one file `test_core.py`; fakes, no network.

## Review Focus

1. **Phone locks / tab closes mid-turn** (e.g. during a 5-min Claude run) → the turn keeps running and the answer is saved; reopening the chat shows it. Test: `test_start_turn_finishes_without_consumer` (Task 6).
2. **Ambiguous project name with full-access Claude** ("bot" matches two projects) → never guess a folder; return an error listing candidates. Test: `test_need_project_ambiguous_and_unknown` (Task 4).
3. **Gemini returns malformed JSON** (promote, sync, extraction) → tool/job reports an error, nothing crashes, bad transcript isn't retried forever. Tests: `test_promote_idea_bad_json_is_error` (Task 4), `test_sync_bad_llm_json_marks_failed_and_synced` (Task 7).
4. **Reminder time not ISO or in the past** ("tomorrow 5pm", yesterday) → clear error Gemini can correct from. Test: `test_parse_when` (Task 4).
5. **Transcript being written right now** (truncated last JSONL line, meta/command lines) → skipped lines, rest parsed. Test: `test_parse_transcript_skips_noise_and_partial_line` (Task 7).

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | deps, uv project |
| `.gitignore` | keep `.env`, db, logs, backups out of git |
| `config.py` | `.env` loading, typed settings, `TZ` |
| `db.py` | schema, connection, tiny query helpers, time helpers |
| `llm.py` | Gemini wrapper: fallback chain, throttle, 429/503/404 handling, embeddings, vector helpers |
| `memory.py` | remember (dedup-merge), recall (vector/keyword), related items |
| `tools.py` | Gemini tool declarations + implementations, project lookup, ranking, resume text |
| `claude_runner.py` | headless Claude Code subprocess |
| `agent.py` | system prompt, chat turn loop, detached turns, greeting, extraction, summaries |
| `jobs.py` | Claude history sync, nightly reflection/recaps/backup, scheduler tick |
| `main.py` | FastAPI app: auth, API routes, SSE, startup |
| `static/login.html`, `static/index.html` | UI |
| `test_core.py` | all tests |
| `README.md` | setup, run, systemd |

---

### Task 1: Scaffold, config, database

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `config.py`, `db.py`, `test_core.py`

**Interfaces:**
- Produces: `config.load_env(path) -> dict`, `config._list(s) -> list[str]`, module constants `GEMINI_API_KEY, MODELS, EMBED_MODEL, SEARCH_MODELS, RPM_LIMIT, APP_PASSWORD, SESSION_SECRET, PORT, TZ, CLAUDE_TIMEOUT, SYNC_MAX_CALLS, DB_PATH, CLAUDE_HISTORY_DIR, ROOT`.
- Produces: `db.connect(path) -> sqlite3.Connection`, `db.now() -> str`, `db.to_utc(dt) -> str`, `db.parse(s) -> datetime`, `db.rows(con, sql, *args) -> list[dict]`, `db.one(con, sql, *args) -> dict|None`, `db.insert(con, table, **fields) -> int`, `db.update(con, table, id, **fields)`, `db.kv_get(con, key, default=None)`, `db.kv_set(con, key, value)`, `db.public(row) -> dict` (drops `embedding`).
- Produces: pytest fixture `con` in `test_core.py`.

- [ ] **Step 1: Init git and project files**

```bash
cd /mnt/newdisk/neel/ai-new-projects
git init
```

`.gitignore`:
```
.env
assistant.db*
assistant.log
backups/
.venv/
__pycache__/
.pytest_cache/
```

`pyproject.toml`:
```toml
[project]
name = "gemini-assistant"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "fastapi>=0.115",
  "uvicorn>=0.30",
  "google-genai>=2.25",
  "httpx>=0.27",
  "itsdangerous>=2.2",
  "numpy>=1.26",
]

[dependency-groups]
dev = ["pytest>=8"]

[tool.uv]
package = false
```

Run: `uv sync` — Expected: creates `.venv`, installs deps.

- [ ] **Step 2: Write failing tests** — create `test_core.py`:

```python
import asyncio
import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import config
import db


@pytest.fixture
def con(tmp_path):
    return db.connect(str(tmp_path / "t.db"))


def test_load_env_lowercase_key_and_comments(tmp_path):
    p = tmp_path / ".env"
    p.write_text('gemini_api_key="abc123"\n# comment\nMODELS=a, b  # inline\n')
    env = config.load_env(p)
    assert env["GEMINI_API_KEY"] == "abc123"
    assert config._list(env["MODELS"]) == ["a", "b"]
    assert env["TIMEZONE"] == "Asia/Kolkata"


def test_db_roundtrip(con):
    pid = db.insert(con, "projects", name="Key Manager", created_at=db.now(), embedding=b"\x00")
    assert db.one(con, "SELECT id FROM projects WHERE name=?", "key manager")["id"] == pid  # NOCASE
    db.update(con, "projects", pid, status="stalled")
    row = db.one(con, "SELECT * FROM projects WHERE id=?", pid)
    assert row["status"] == "stalled"
    assert "embedding" not in db.public(row)
    db.kv_set(con, "a", "1")
    db.kv_set(con, "a", "2")
    assert db.kv_get(con, "a") == "2" and db.kv_get(con, "missing", "d") == "d"


def test_time_helpers_are_utc():
    assert db.now().endswith("+00:00")
    ist = datetime(2026, 9, 29, 15, 18, tzinfo=config.TZ)
    assert db.to_utc(ist) == "2026-09-29T09:48:00+00:00"
    assert db.parse("2026-09-10T06:15:41.731Z").tzinfo is not None
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest -q test_core.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'config'`

- [ ] **Step 4: Implement `config.py`**

```python
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
    "CLAUDE_TIMEOUT": "600",
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
SYNC_MAX_CALLS = int(_env["SYNC_MAX_CALLS"])
DB_PATH = _env["DB_PATH"]
CLAUDE_HISTORY_DIR = Path(_env["CLAUDE_HISTORY_DIR"])
```

- [ ] **Step 5: Implement `db.py`**

```python
"""SQLite schema and small helpers. All timestamps are UTC ISO strings."""
import sqlite3
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations(
  id INTEGER PRIMARY KEY, title TEXT DEFAULT 'New chat', summary TEXT DEFAULT '',
  summary_upto INTEGER DEFAULT 0, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS messages(
  id INTEGER PRIMARY KEY,
  conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  role TEXT NOT NULL, content TEXT NOT NULL, meta TEXT DEFAULT '{}', created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS projects(
  id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE COLLATE NOCASE, path TEXT, main_files TEXT,
  goal TEXT DEFAULT '', status TEXT DEFAULT 'active', next_step TEXT DEFAULT '',
  notes TEXT DEFAULT '', last_activity_at TEXT, claude_session_id TEXT, embedding BLOB,
  created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS memories(
  id INTEGER PRIMARY KEY, text TEXT NOT NULL, kind TEXT DEFAULT 'note', source TEXT DEFAULT 'chat',
  project_id INTEGER REFERENCES projects(id) ON DELETE SET NULL, embedding BLOB,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS ideas(
  id INTEGER PRIMARY KEY, text TEXT NOT NULL, tags TEXT DEFAULT '', status TEXT DEFAULT 'inbox',
  project_id INTEGER REFERENCES projects(id) ON DELETE SET NULL, embedding BLOB,
  created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS reminders(
  id INTEGER PRIMARY KEY, text TEXT NOT NULL, due_at TEXT NOT NULL, done INTEGER DEFAULT 0,
  project_id INTEGER REFERENCES projects(id) ON DELETE SET NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS decisions(
  id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  text TEXT NOT NULL, why TEXT DEFAULT '', source TEXT DEFAULT 'chat', decided_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS links(
  id INTEGER PRIMARY KEY, url TEXT NOT NULL, title TEXT DEFAULT '', summary TEXT DEFAULT '',
  project_id INTEGER REFERENCES projects(id) ON DELETE SET NULL,
  idea_id INTEGER REFERENCES ideas(id) ON DELETE SET NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS recaps(
  id INTEGER PRIMARY KEY, period TEXT NOT NULL, period_start TEXT NOT NULL,
  content TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS claude_runs(
  id INTEGER PRIMARY KEY, project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
  prompt TEXT NOT NULL, output TEXT DEFAULT '', session_id TEXT, files_changed TEXT DEFAULT '[]',
  exit_code INTEGER, duration_s REAL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sync_state(
  file_path TEXT PRIMARY KEY, sha256 TEXT NOT NULL, synced_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS model_state(model TEXT PRIMARY KEY, exhausted_until REAL NOT NULL);
CREATE TABLE IF NOT EXISTS kv(key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def to_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def connect(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    con.executescript(SCHEMA)
    return con


def rows(con, sql: str, *args) -> list[dict]:
    return [dict(r) for r in con.execute(sql, args)]


def one(con, sql: str, *args) -> dict | None:
    r = con.execute(sql, args).fetchone()
    return dict(r) if r else None


# Table/column names below always come from code, never from request bodies.
def insert(con, table: str, **fields) -> int:
    cols = ",".join(fields)
    marks = ",".join("?" * len(fields))
    return con.execute(f"INSERT INTO {table}({cols}) VALUES({marks})", tuple(fields.values())).lastrowid


def update(con, table: str, id_: int, **fields) -> None:
    if fields:
        sets = ",".join(f"{k}=?" for k in fields)
        con.execute(f"UPDATE {table} SET {sets} WHERE id=?", (*fields.values(), id_))


def kv_get(con, key: str, default=None):
    r = one(con, "SELECT value FROM kv WHERE key=?", key)
    return r["value"] if r else default


def kv_set(con, key: str, value: str) -> None:
    con.execute("INSERT INTO kv(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))


def public(row: dict) -> dict:
    return {k: v for k, v in row.items() if k != "embedding"}
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest -q test_core.py`
Expected: 3 passed

- [ ] **Step 7: Commit**

```bash
git add .gitignore pyproject.toml uv.lock config.py db.py test_core.py docs/
git commit -m "feat: scaffold config and sqlite schema"
```

---

### Task 2: Gemini wrapper with fallback chain

**Files:**
- Create: `llm.py`
- Test: `test_core.py` (append)

**Interfaces:**
- Consumes: `db.one`, `con` (table `model_state`).
- Produces: `llm.AllModelsExhausted(until: float)` with `.until`; `llm.classify(err) -> "day"|"minute"|"dead"|"fatal"`; `llm.next_pacific_midnight(ts) -> float`; `llm.pack(vec) -> bytes`; `llm.unpack(b) -> np.ndarray`; `llm.top_k(query_vec, items: list[tuple[key, bytes]], k) -> list[tuple[key, float]]`.
- Produces: `llm.LLM(client, con, models, embed_model, search_models, rpm_limit, clock=time.time)` with
  `async generate(contents, config=None, background=False) -> (response, model)`,
  `async search(query) -> (response, model)`,
  `async embed(texts) -> list[np.ndarray] | None`,
  `exhausted_until(model) -> float`, attrs `models`, `dead`, `last_model`.

- [ ] **Step 1: Append failing tests**

```python
from google.genai import errors

import llm
from llm import LLM, AllModelsExhausted


def api_err(code, msg="quota"):
    body = {"error": {"code": code, "message": msg, "status": "X"}}
    return errors.ClientError(code, body) if code < 500 else errors.ServerError(code, body)


class FakeModels:
    def __init__(self, script):
        self.script, self.seen = script, []

    async def generate_content(self, model, contents, config=None):
        self.seen.append(model)
        r = self.script.get(model, "ok")
        if isinstance(r, Exception):
            raise r
        return SimpleNamespace(text=r)

    async def embed_content(self, model, contents):
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[1.0, 0.0]) for _ in contents])


def make_llm(con, script, t, rpm=15):
    client = SimpleNamespace(aio=SimpleNamespace(models=FakeModels(script)))
    return LLM(client, con, ["a", "b", "c"], "emb", ["s"], rpm, clock=lambda: t[0])


def test_minute_429_falls_back_and_recovers(con):
    t = [1000.0]
    m = make_llm(con, {"a": api_err(429, "rate limit per minute")}, t)
    resp, model = asyncio.run(m.generate("hi"))
    assert model == "b" and m.last_model == "b"
    assert m.exhausted_until("a") == 1060.0
    m.client.aio.models.script = {}
    t[0] = 1061.0
    assert asyncio.run(m.generate("hi"))[1] == "a"


def test_day_429_until_pacific_midnight(con):
    t = [datetime(2026, 9, 29, 9, 48, tzinfo=timezone.utc).timestamp()]
    m = make_llm(con, {"a": api_err(429, "GenerateRequestsPerDayPerProjectPerModel-FreeTier")}, t)
    asyncio.run(m.generate("hi"))
    # 02:48 PDT -> next midnight PDT = 07:00 UTC next day
    assert m.exhausted_until("a") == datetime(2026, 9, 30, 7, 0, tzinfo=timezone.utc).timestamp()


def test_503_and_404_skip(con):
    t = [0.0]
    m = make_llm(con, {"a": api_err(503), "b": api_err(404)}, t)
    assert asyncio.run(m.generate("hi"))[1] == "c"
    assert "b" in m.dead and m.exhausted_until("a") == 60.0


def test_all_exhausted_raises_with_earliest_time(con):
    t = [0.0]
    m = make_llm(con, {k: api_err(429) for k in "abc"}, t)
    with pytest.raises(AllModelsExhausted) as e:
        asyncio.run(m.generate("hi"))
    assert e.value.until == 60.0


def test_fatal_400_raises(con):
    m = make_llm(con, {"a": api_err(400, "bad request")}, [0.0])
    with pytest.raises(errors.ClientError):
        asyncio.run(m.generate("hi"))


def test_local_rpm_throttle_moves_to_next_model(con):
    m = make_llm(con, {}, [0.0], rpm=2)
    models = [asyncio.run(m.generate("x"))[1] for _ in range(3)]
    assert models == ["a", "a", "b"]


def test_background_uses_only_first_model(con):
    m = make_llm(con, {"a": api_err(429)}, [0.0])
    with pytest.raises(AllModelsExhausted):
        asyncio.run(m.generate("x", background=True))


def test_top_k_orders_by_cosine():
    items = [(1, llm.pack(np.array([1, 0], np.float32))), (2, llm.pack(np.array([0, 1], np.float32)))]
    hits = llm.top_k(np.array([0.9, 0.1], np.float32), items, 2)
    assert [h[0] for h in hits] == [1, 2] and hits[0][1] > 0.9
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest -q test_core.py -k "429 or 503 or exhausted or fatal or throttle or background or top_k"`
Expected: FAIL — `No module named 'llm'`

- [ ] **Step 3: Implement `llm.py`**

```python
"""Gemini wrapper: ordered model fallback, local RPM throttle, quota bookkeeping, embeddings."""
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
from google.genai import errors, types

import db

PACIFIC = ZoneInfo("America/Los_Angeles")  # Gemini daily quotas reset at midnight Pacific


class AllModelsExhausted(Exception):
    def __init__(self, until: float):
        super().__init__(f"all models out of quota until {until}")
        self.until = until


def next_pacific_midnight(ts: float) -> float:
    d = datetime.fromtimestamp(ts, PACIFIC)
    return (datetime(d.year, d.month, d.day, tzinfo=PACIFIC) + timedelta(days=1)).timestamp()


def classify(err: errors.APIError) -> str:
    if err.code == 429:
        return "day" if "PerDay" in str(err) else "minute"
    if err.code == 404:
        return "dead"
    if err.code >= 500:
        return "minute"  # 503 overloaded etc: skip this model for a minute
    return "fatal"


def pack(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def unpack(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def top_k(query: np.ndarray, items: list[tuple], k: int) -> list[tuple]:
    """items: [(key, packed_vector)] -> [(key, cosine)] best first."""
    if not items:
        return []
    mat = np.stack([unpack(b) for _, b in items])
    sims = mat @ query / (np.linalg.norm(mat, axis=1) * np.linalg.norm(query) + 1e-9)
    return [(items[i][0], float(sims[i])) for i in np.argsort(-sims)[:k]]


class LLM:
    def __init__(self, client, con, models, embed_model, search_models, rpm_limit, clock=time.time):
        self.client, self.con = client, con
        self.models, self.embed_model, self.search_models = models, embed_model, search_models
        self.rpm_limit, self.clock = rpm_limit, clock
        self.calls: dict[str, deque] = defaultdict(deque)
        self.dead: set[str] = set()
        self.last_model: str | None = None

    def exhausted_until(self, model: str) -> float:
        r = db.one(self.con, "SELECT exhausted_until FROM model_state WHERE model=?", model)
        return r["exhausted_until"] if r else 0.0

    def _mark(self, model: str, until: float) -> None:
        self.con.execute("INSERT INTO model_state(model, exhausted_until) VALUES(?, ?) "
                         "ON CONFLICT(model) DO UPDATE SET exhausted_until=excluded.exhausted_until",
                         (model, until))

    def _usable(self, model: str) -> bool:
        now = self.clock()
        if model in self.dead or self.exhausted_until(model) > now:
            return False
        q = self.calls[model]
        while q and q[0] <= now - 60:
            q.popleft()
        # ponytail: at the local RPM cap we move to the next model instead of waiting
        return len(q) < self.rpm_limit

    def _fail(self, model: str, err: errors.APIError) -> None:
        kind = classify(err)
        if kind == "fatal":
            raise err
        if kind == "dead":
            self.dead.add(model)
        elif kind == "day":
            self._mark(model, next_pacific_midnight(self.clock()))
        else:
            self._mark(model, self.clock() + 60)

    async def _call(self, chain: list[str], fn):
        for model in chain:
            if not self._usable(model):
                continue
            self.calls[model].append(self.clock())
            try:
                return await fn(model), model
            except errors.APIError as e:
                self._fail(model, e)
        now = self.clock()
        until = min((self.exhausted_until(m) for m in chain if m not in self.dead), default=0.0)
        raise AllModelsExhausted(until if until > now else now + 60)

    async def generate(self, contents, config=None, background=False):
        chain = self.models[:1] if background else self.models  # jobs never eat the chat quota
        resp, model = await self._call(chain, lambda m: self.client.aio.models.generate_content(
            model=m, contents=contents, config=config))
        self.last_model = model
        return resp, model

    async def search(self, query: str):
        cfg = types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())])
        return await self._call(self.search_models, lambda m: self.client.aio.models.generate_content(
            model=m, contents=query, config=cfg))

    async def embed(self, texts: list[str]) -> list[np.ndarray] | None:
        """None when the embedding model is unavailable; callers fall back to keywords."""
        try:
            resp, _ = await self._call([self.embed_model], lambda m: self.client.aio.models.embed_content(
                model=m, contents=texts))
        except AllModelsExhausted:
            return None
        return [np.asarray(e.values, dtype=np.float32) for e in resp.embeddings]
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest -q test_core.py`
Expected: all pass (11)

- [ ] **Step 5: Commit**

```bash
git add llm.py test_core.py
git commit -m "feat: gemini wrapper with quota-aware model fallback"
```

---

### Task 3: Memory (remember, recall, related)

**Files:**
- Create: `memory.py`
- Test: `test_core.py` (append — also defines shared `FakeLLM` used by Tasks 4, 6, 7, 8)

**Interfaces:**
- Consumes: `db.*`, `llm.pack`, `llm.top_k`, an object with `async embed(texts)`.
- Produces: `memory.DUP = 0.9`, `memory.RELATED = 0.75`, `memory.RECALL_MIN = 0.55`;
  `async memory.remember(con, llm, text, kind="note", source="chat", project_id=None) -> {"id": int, "merged": bool}`;
  `async memory.recall(con, llm, query, k=8) -> list[{"id","text","kind","source","project_id",("score")}]`;
  `memory.keyword_search(con, query, k=8) -> list[dict]`;
  `memory.related(con, vec, exclude_idea=None) -> list[{"type": "idea"|"project", "id", "text", "score"}]`.
- Produces (tests): `FakeLLM(vecs=None, replies=None)` with `embed`, `generate`, `prompts`, `last_model`, `models`, `dead`, `exhausted_until`.

- [ ] **Step 1: Append failing tests**

```python
import memory


class FakeLLM:
    """Stand-in for llm.LLM. vecs: text -> vector (any unknown text => embed returns None).
    replies: queue for generate(); str -> response with .text, Exception -> raised, else returned as-is."""

    def __init__(self, vecs=None, replies=None):
        self.vecs, self.replies, self.prompts = vecs or {}, list(replies or []), []
        self.last_model, self.models, self.dead = "fake", ["fake"], set()

    def exhausted_until(self, model):
        return 0.0

    async def embed(self, texts):
        if not all(t in self.vecs for t in texts):
            return None
        return [np.asarray(self.vecs[t], np.float32) for t in texts]

    async def generate(self, contents, config=None, background=False):
        self.prompts.append(contents)
        r = self.replies.pop(0) if self.replies else '{"memories": []}'
        if isinstance(r, Exception):
            raise r
        if isinstance(r, str):
            r = SimpleNamespace(text=r, function_calls=None, candidates=[])
        return r, "fake"


def test_remember_merges_near_duplicate(con):
    fl = FakeLLM({"uses Qwen3 4B": [1, 0], "uses Qwen3 4B now": [0.99, 0.05], "likes tea": [0, 1]})
    a = asyncio.run(memory.remember(con, fl, "uses Qwen3 4B"))
    b = asyncio.run(memory.remember(con, fl, "uses Qwen3 4B now"))
    c = asyncio.run(memory.remember(con, fl, "likes tea", kind="preference"))
    assert b == {"id": a["id"], "merged": True} and c["merged"] is False
    assert [m["text"] for m in db.rows(con, "SELECT text FROM memories ORDER BY id")] == ["uses Qwen3 4B now", "likes tea"]


def test_recall_vector_and_keyword_fallback(con):
    fl = FakeLLM({"deploys on runpod": [1, 0], "cloud gpu?": [0.9, 0.1]})
    asyncio.run(memory.remember(con, fl, "deploys on runpod"))
    assert asyncio.run(memory.recall(con, fl, "cloud gpu?"))[0]["text"] == "deploys on runpod"
    no_embed = FakeLLM()
    asyncio.run(memory.remember(con, no_embed, "prefers FastAPI over Flask"))
    assert db.one(con, "SELECT embedding FROM memories WHERE text LIKE 'prefers%'")["embedding"] is None
    assert asyncio.run(memory.recall(con, no_embed, "which fastapi thing"))[0]["text"].startswith("prefers")


def test_related_finds_similar_ideas_and_projects(con):
    db.insert(con, "projects", name="distil", embedding=llm.pack(np.array([1, 0], np.float32)), created_at=db.now())
    db.insert(con, "ideas", text="far away", embedding=llm.pack(np.array([0, 1], np.float32)), created_at=db.now())
    rel = memory.related(con, np.array([0.95, 0.05], np.float32))
    assert [(r["type"], r["text"]) for r in rel] == [("project", "distil")]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest -q test_core.py -k "remember or recall or related"`
Expected: FAIL — `No module named 'memory'`

- [ ] **Step 3: Implement `memory.py`**

```python
"""Long-term memory: dedup-merging writes, vector recall with keyword fallback, related items."""
import re

import numpy as np

import db
from llm import pack, top_k

DUP = 0.9          # cosine at/above which two memories are "the same" -> merge
RELATED = 0.75     # threshold for "this relates to that" suggestions
RECALL_MIN = 0.55  # ponytail: tuning knob; raise if recall pulls in noise, lower if it misses things


def _vectors(con, table: str) -> list[tuple]:
    return [(r["id"], r["embedding"]) for r in
            db.rows(con, f"SELECT id, embedding FROM {table} WHERE embedding IS NOT NULL")]


async def remember(con, llm, text, kind="note", source="chat", project_id=None) -> dict:
    vecs = await llm.embed([text])
    vec = vecs[0] if vecs else None
    if vec is not None:
        hit = top_k(vec, _vectors(con, "memories"), 1)
        if hit and hit[0][1] >= DUP:
            mid = hit[0][0]
            db.update(con, "memories", mid, text=text, kind=kind, embedding=pack(vec), updated_at=db.now())
            return {"id": mid, "merged": True}
    mid = db.insert(con, "memories", text=text, kind=kind, source=source, project_id=project_id,
                    embedding=pack(vec) if vec is not None else None,
                    created_at=db.now(), updated_at=db.now())
    return {"id": mid, "merged": False}


def keyword_search(con, query: str, k: int = 8) -> list[dict]:
    words = re.findall(r"\w{3,}", query.lower())[:8]
    if not words:
        return []
    where = " OR ".join("lower(text) LIKE ?" for _ in words)
    return db.rows(con, f"SELECT id, text, kind, source, project_id FROM memories WHERE {where} "
                        "ORDER BY updated_at DESC LIMIT ?", *[f"%{w}%" for w in words], k)


async def recall(con, llm, query: str, k: int = 8) -> list[dict]:
    vecs = await llm.embed([query])
    if vecs is None:
        return keyword_search(con, query, k)
    out = []
    for mid, score in top_k(vecs[0], _vectors(con, "memories"), k):
        if score >= RECALL_MIN:
            m = db.one(con, "SELECT id, text, kind, source, project_id FROM memories WHERE id=?", mid)
            m["score"] = round(score, 3)
            out.append(m)
    return out


def related(con, vec: np.ndarray, exclude_idea: int | None = None) -> list[dict]:
    items = [(("idea", i), b) for i, b in _vectors(con, "ideas") if i != exclude_idea]
    items += [(("project", i), b) for i, b in _vectors(con, "projects")]
    out = []
    for (kind, id_), score in top_k(vec, items, 3):
        if score < RELATED:
            continue
        col = "text" if kind == "idea" else "name"
        row = db.one(con, f"SELECT {col} AS text FROM {kind}s WHERE id=?", id_)
        out.append({"type": kind, "id": id_, "text": row["text"], "score": round(score, 3)})
    return out
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest -q test_core.py`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add memory.py test_core.py
git commit -m "feat: long-term memory with dedup merge and recall"
```

---

### Task 4: Agent tools

**Files:**
- Create: `tools.py`
- Test: `test_core.py` (append)

**Interfaces:**
- Consumes: `memory.*`, `db.*`, `llm.pack`, `config.TZ`; `ctx.runner.run(project: dict, prompt: str, follow_up: bool) -> dict` (Task 5); `ctx.llm.generate/embed/search`.
- Produces: `tools.Ctx(con, llm, runner, conversation_id=None)`; `tools.ToolError`; `tools.TOOLS: dict[name, (fn, FunctionDeclaration)]`; `tools.declarations() -> list[types.Tool]`; `async tools.run_tool(ctx, name, args) -> dict` (never raises; errors → `{"error": str}`);
  `tools.norm(s)`, `tools.find_projects(con, name) -> list[dict]`, `tools.find_project(con, name) -> dict|None`, `tools.need_project(con, name) -> dict` (raises ToolError);
  `tools.parse_when(when, now=None) -> datetime (UTC)`; `tools.fmt_local(iso) -> str`; `tools.check_fields(fields)` (raises ToolError on bad status/missing folder); `tools.STATUSES`;
  `async tools.embed_project(con, llm, pid)`; `tools.rank_projects(con, now=None) -> list[{"id","name","next_step","days_idle","score","reasons"}]`; `tools.build_resume(con, project) -> str`; `tools.html_to_text(raw) -> str`.
- Tool names (Gemini-callable, also used by routes): `save_memory, search_memory, create_project, update_project, list_projects, save_idea, list_ideas, promote_idea(idea_id), set_reminder, list_reminders, complete_reminder(reminder_id), log_decision, list_decisions, capture_link, get_recap, what_to_work_on, resume_context, ask_claude_code(project, prompt, follow_up=False), web_search`.

- [ ] **Step 1: Append failing tests**

```python
import tools


def ctx_for(con, fl=None, runner=None, cid=None):
    return tools.Ctx(con, fl or FakeLLM(), runner, cid)


def test_parse_when():
    now = datetime(2026, 9, 29, 9, 0, tzinfo=timezone.utc)  # 14:30 IST
    assert tools.parse_when("2026-09-29T17:00", now) == datetime(2026, 9, 29, 11, 30, tzinfo=timezone.utc)
    for bad in ["tomorrow 5pm", "2026-09-29T10:00"]:  # not ISO / already past in IST
        with pytest.raises(tools.ToolError):
            tools.parse_when(bad, now)


def test_find_project_fuzzy(con):
    db.insert(con, "projects", name="ai_call_bot", created_at=db.now())
    assert tools.find_project(con, "AI call bot")["name"] == "ai_call_bot"
    assert tools.find_project(con, "nothing") is None


def test_need_project_ambiguous_and_unknown(con):
    db.insert(con, "projects", name="patent-bot-DB-backend", created_at=db.now())
    db.insert(con, "projects", name="ai-call-bot", created_at=db.now())
    with pytest.raises(tools.ToolError, match="ambiguous"):
        tools.need_project(con, "bot")
    with pytest.raises(tools.ToolError, match="Known projects"):
        tools.need_project(con, "zzz")


def test_run_tool_errors_never_raise(con):
    c = ctx_for(con)
    assert "unknown tool" in asyncio.run(tools.run_tool(c, "nope", {}))["error"]
    assert "bad arguments" in asyncio.run(tools.run_tool(c, "set_reminder", {"text": "x"}))["error"]
    assert "not an ISO" in asyncio.run(tools.run_tool(c, "set_reminder", {"text": "x", "when": "friday"}))["error"]


def test_save_idea_and_promote(con):
    fl = FakeLLM(replies=['{"name": "GPU Tracker", "goal": "Track GPU prices", "steps": ["scrape", "alert"]}'])
    c = ctx_for(con, fl)
    iid = asyncio.run(tools.run_tool(c, "save_idea", {"text": "track gpu prices"}))["id"]
    r = asyncio.run(tools.run_tool(c, "promote_idea", {"idea_id": iid}))
    p = db.one(con, "SELECT * FROM projects WHERE id=?", r["project_id"])
    assert p["name"] == "GPU Tracker" and p["next_step"] == "scrape" and "- alert" in p["notes"]
    assert db.one(con, "SELECT status FROM ideas WHERE id=?", iid)["status"] == "promoted"


def test_promote_idea_bad_json_is_error(con):
    c = ctx_for(con, FakeLLM(replies=["not json"]))
    iid = asyncio.run(tools.run_tool(c, "save_idea", {"text": "x"}))["id"]
    assert "error" in asyncio.run(tools.run_tool(c, "promote_idea", {"idea_id": iid}))
    assert db.one(con, "SELECT status FROM ideas WHERE id=?", iid)["status"] == "inbox"


def test_rank_projects(con):
    now = datetime(2026, 9, 29, tzinfo=timezone.utc)
    a = db.insert(con, "projects", name="A", last_activity_at=db.to_utc(now - timedelta(days=1)), created_at=db.now())
    b = db.insert(con, "projects", name="B", status="stalled", last_activity_at=db.to_utc(now - timedelta(days=10)), created_at=db.now())
    db.insert(con, "projects", name="C", status="done", last_activity_at=db.to_utc(now), created_at=db.now())
    db.insert(con, "reminders", text="ship B", due_at=db.to_utc(now + timedelta(hours=5)), project_id=b, created_at=db.now())
    ranked = tools.rank_projects(con, now)
    assert [p["name"] for p in ranked] == ["B", "A"]
    assert any("reminder" in r for r in ranked[0]["reasons"])


def test_build_resume(con):
    pid = db.insert(con, "projects", name="key_manager", path="/tmp", goal="Rotate API keys", next_step="add tests", created_at=db.now())
    db.insert(con, "decisions", project_id=pid, text="Use SQLite", why="single user", decided_at=db.now())
    text = tools.build_resume(con, db.one(con, "SELECT * FROM projects WHERE id=?", pid))
    for s in ["key_manager", "`/tmp`", "Rotate API keys", "add tests", "Use SQLite (why: single user)"]:
        assert s in text


def test_ask_claude_code_requires_path(con):
    db.insert(con, "projects", name="nopath", created_at=db.now())
    r = asyncio.run(tools.run_tool(ctx_for(con), "ask_claude_code", {"project": "nopath", "prompt": "hi"}))
    assert "no folder path" in r["error"]


def test_html_to_text():
    assert tools.html_to_text("<html><script>x()</script><p>Hi &amp; bye</p></html>") == "Hi & bye"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest -q test_core.py -k "parse_when or project or run_tool or idea or rank or resume or claude_code or html"`
Expected: FAIL — `No module named 'tools'`

- [ ] **Step 3: Implement `tools.py`**

```python
"""Gemini-callable tools. Each returns a JSON-able dict; failures become {"error": ...} for Gemini."""
import html
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from google.genai import types

import config
import db
import memory
from llm import pack

log = logging.getLogger("tools")
S = {"type": "string"}
I = {"type": "integer"}
STATUSES = ["active", "stalled", "paused", "done"]
PROJECT_FIELDS = {
    "goal": S,
    "status": {"type": "string", "enum": STATUSES},
    "next_step": S,
    "notes": S,
    "path": {"type": "string", "description": "absolute folder path on this machine"},
    "main_files": {"type": "string", "description": "comma-separated key files"},
}


class ToolError(Exception):
    pass


@dataclass
class Ctx:
    con: object
    llm: object
    runner: object
    conversation_id: int | None = None


TOOLS: dict[str, tuple] = {}


def tool(description: str, params: dict | None = None, required=()):
    def deco(fn):
        TOOLS[fn.__name__] = (fn, types.FunctionDeclaration(
            name=fn.__name__, description=description,
            parameters_json_schema={"type": "object", "properties": params or {}, "required": list(required)}))
        return fn
    return deco


def declarations() -> list[types.Tool]:
    return [types.Tool(function_declarations=[d for _, d in TOOLS.values()])]


async def run_tool(ctx: Ctx, name: str, args: dict) -> dict:
    if name not in TOOLS:
        return {"error": f"unknown tool {name}"}
    try:
        return await TOOLS[name][0](ctx, **args)
    except ToolError as e:
        return {"error": str(e)}
    except TypeError as e:
        return {"error": f"bad arguments for {name}: {e}"}
    except Exception as e:
        log.exception("tool %s failed", name)
        return {"error": f"{type(e).__name__}: {e}"}


# ---------- helpers ----------

def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def find_projects(con, name: str) -> list[dict]:
    n = norm(name)
    if not n:
        return []
    ps = db.rows(con, "SELECT * FROM projects")
    exact = [p for p in ps if norm(p["name"]) == n]
    return exact or [p for p in ps if n in norm(p["name"]) or norm(p["name"]) in n]


def find_project(con, name: str | None) -> dict | None:
    ps = find_projects(con, name) if name else []
    return ps[0] if len(ps) == 1 else None


def need_project(con, name: str) -> dict:
    ps = find_projects(con, name)
    if len(ps) == 1:
        return ps[0]
    if ps:
        raise ToolError(f"'{name}' is ambiguous: {', '.join(p['name'] for p in ps)}. Ask the user which one.")
    known = ", ".join(p["name"] for p in db.rows(con, "SELECT name FROM projects ORDER BY name"))
    raise ToolError(f"No project matching '{name}'. Known projects: {known or 'none'}.")


def parse_when(when: str, now: datetime | None = None) -> datetime:
    """ISO datetime from Gemini; naive means local TZ. Returns aware UTC."""
    try:
        dt = datetime.fromisoformat(when.strip().replace("Z", "+00:00"))
    except ValueError:
        raise ToolError(f"'{when}' is not an ISO datetime like 2026-10-02T17:00 (local time)")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=config.TZ)
    if dt <= (now or datetime.now(timezone.utc)):
        raise ToolError(f"{dt.astimezone(config.TZ):%Y-%m-%d %H:%M} is in the past")
    return dt.astimezone(timezone.utc)


def fmt_local(iso: str | None) -> str:
    return db.parse(iso).astimezone(config.TZ).strftime("%a %d %b %Y %H:%M") if iso else "-"


def check_fields(fields: dict) -> None:
    if fields.get("status") and fields["status"] not in STATUSES:
        raise ToolError(f"status must be one of {STATUSES}")
    if fields.get("path") and not Path(fields["path"]).expanduser().is_dir():
        raise ToolError(f"folder {fields['path']} does not exist on this machine")


async def embed_project(con, llm, pid: int) -> None:
    p = db.one(con, "SELECT name, goal FROM projects WHERE id=?", pid)
    vecs = await llm.embed([f"{p['name']}: {p['goal'] or ''}"])
    if vecs:
        db.update(con, "projects", pid, embedding=pack(vecs[0]))


def html_to_text(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style|noscript)\b.*?</\1>", " ", raw)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", raw))).strip()


def rank_projects(con, now: datetime | None = None) -> list[dict]:
    # ponytail: fixed-weight heuristic (due reminders > momentum > stalled); tune weights here
    now = now or datetime.now(timezone.utc)
    due = {r["project_id"]: r["n"] for r in db.rows(
        con, "SELECT project_id, COUNT(*) n FROM reminders WHERE done=0 AND due_at <= ? "
             "AND project_id IS NOT NULL GROUP BY project_id", db.to_utc(now + timedelta(days=2)))}
    out = []
    for p in db.rows(con, "SELECT id, name, next_step, last_activity_at FROM projects "
                          "WHERE status IN ('active', 'stalled')"):
        idle = (now - db.parse(p["last_activity_at"])).days if p["last_activity_at"] else 30
        n = due.get(p["id"], 0)
        reasons = []
        if n:
            reasons.append(f"{n} reminder(s) due within 2 days")
        if idle <= 3:
            reasons.append("momentum: worked on in the last 3 days")
        elif 7 <= idle <= 30:
            reasons.append(f"stalled for {idle} days")
        score = 3 * n + (2 if idle <= 3 else 0) + (1 if 7 <= idle <= 30 else 0)
        out.append({"id": p["id"], "name": p["name"], "next_step": p["next_step"],
                    "days_idle": idle, "score": score, "reasons": reasons})
    return sorted(out, key=lambda x: -x["score"])[:5]


def build_resume(con, p: dict) -> str:
    decisions = db.rows(con, "SELECT text, why FROM decisions WHERE project_id=? "
                             "ORDER BY decided_at DESC LIMIT 8", p["id"])
    mems = db.rows(con, "SELECT text FROM memories WHERE project_id=? ORDER BY updated_at DESC LIMIT 10", p["id"])
    where = f" at `{p['path']}`" if p["path"] else ""
    lines = [f"I'm resuming work on the project **{p['name']}**{where}.", "",
             f"Goal: {p['goal'] or '(not set)'}",
             f"Status: {p['status']}",
             f"Next step: {p['next_step'] or '(not set)'}"]
    if p.get("main_files"):
        lines.append(f"Key files: {p['main_files']}")
    if p["notes"]:
        lines += ["", "Notes:", p["notes"]]
    if decisions:
        lines += ["", "Key decisions so far:"]
        lines += [f"- {d['text']}" + (f" (why: {d['why']})" if d["why"] else "") for d in decisions]
    if mems:
        lines += ["", "Things to remember:"] + [f"- {m['text']}" for m in mems]
    lines += ["", "Read the relevant files first, confirm the current state in a few lines, "
                  "then continue with the next step."]
    return "\n".join(lines)


# ---------- tools ----------

@tool("Save a durable long-term memory about the user or their work.",
      {"text": S, "kind": {"type": "string", "enum": ["fact", "preference", "decision", "note"]}, "project": S},
      ["text"])
async def save_memory(ctx, text, kind="note", project=None):
    p = find_project(ctx.con, project)
    return await memory.remember(ctx.con, ctx.llm, text, kind, "chat", p["id"] if p else None)


@tool("Search long-term memory by meaning.", {"query": S}, ["query"])
async def search_memory(ctx, query):
    return {"memories": await memory.recall(ctx.con, ctx.llm, query)}


@tool("Create a new project.", {"name": S, **PROJECT_FIELDS}, ["name"])
async def create_project(ctx, name, **fields):
    if db.one(ctx.con, "SELECT id FROM projects WHERE name=?", name):
        raise ToolError(f"project '{name}' already exists")
    fields = {k: v for k, v in fields.items() if k in PROJECT_FIELDS}
    check_fields(fields)
    pid = db.insert(ctx.con, "projects", name=name, created_at=db.now(), last_activity_at=db.now(), **fields)
    await embed_project(ctx.con, ctx.llm, pid)
    return {"id": pid, "created": name}


@tool("Update fields of an existing project (e.g. set its folder path, status, next step).",
      {"name": S, **PROJECT_FIELDS}, ["name"])
async def update_project(ctx, name, **fields):
    p = need_project(ctx.con, name)
    fields = {k: v for k, v in fields.items() if k in PROJECT_FIELDS}
    check_fields(fields)
    db.update(ctx.con, "projects", p["id"], last_activity_at=db.now(), **fields)
    if "goal" in fields:
        await embed_project(ctx.con, ctx.llm, p["id"])
    return {"updated": p["name"], "fields": list(fields)}


@tool("List projects, optionally filtered by status.", {"status": {"type": "string", "enum": STATUSES}})
async def list_projects(ctx, status=None):
    sql = "SELECT name, status, goal, next_step, last_activity_at, path FROM projects"
    ps = db.rows(ctx.con, sql + " WHERE status=? ORDER BY last_activity_at DESC", status) if status \
        else db.rows(ctx.con, sql + " ORDER BY last_activity_at DESC")
    return {"projects": ps}


@tool("Save an idea to the ideas inbox. Returns related ideas/projects.", {"text": S, "tags": S}, ["text"])
async def save_idea(ctx, text, tags=""):
    vecs = await ctx.llm.embed([text])
    vec = vecs[0] if vecs else None
    iid = db.insert(ctx.con, "ideas", text=text, tags=tags, created_at=db.now(),
                    embedding=pack(vec) if vec is not None else None)
    return {"id": iid, "related": memory.related(ctx.con, vec, exclude_idea=iid) if vec is not None else []}


@tool("List ideas by status.", {"status": {"type": "string", "enum": ["inbox", "promoted", "dropped"]}})
async def list_ideas(ctx, status="inbox"):
    return {"ideas": db.rows(ctx.con, "SELECT id, text, tags, created_at FROM ideas WHERE status=? "
                                      "ORDER BY created_at DESC LIMIT 50", status)}


@tool("Promote an inbox idea to a project with a drafted goal and first steps.", {"idea_id": I}, ["idea_id"])
async def promote_idea(ctx, idea_id):
    idea = db.one(ctx.con, "SELECT * FROM ideas WHERE id=?", idea_id)
    if not idea or idea["status"] != "inbox":
        raise ToolError(f"idea {idea_id} not found in inbox")
    resp, _ = await ctx.llm.generate(
        f"Turn this idea into a project plan.\nIdea: {idea['text']}\n"
        'Return JSON: {"name": "short project name", "goal": "one sentence", "steps": ["3-5 concrete first steps"]}',
        types.GenerateContentConfig(response_mime_type="application/json"))
    try:
        plan = json.loads(resp.text)
        name, goal, steps = plan["name"].strip(), plan["goal"], [str(s) for s in plan["steps"]]
    except (ValueError, KeyError, TypeError, AttributeError):
        raise ToolError("could not draft a plan (model returned invalid JSON); try again")
    if db.one(ctx.con, "SELECT id FROM projects WHERE name=?", name):
        name = f"{name} ({idea_id})"
    pid = db.insert(ctx.con, "projects", name=name, goal=goal, next_step=steps[0] if steps else "",
                    notes="First steps:\n" + "\n".join(f"- {s}" for s in steps),
                    last_activity_at=db.now(), created_at=db.now())
    db.update(ctx.con, "ideas", idea_id, status="promoted", project_id=pid)
    await embed_project(ctx.con, ctx.llm, pid)
    return {"project_id": pid, "name": name, "goal": goal, "steps": steps}


@tool("Set a reminder. `when` = ISO local datetime (user's timezone), e.g. 2026-10-02T17:00; "
      "convert natural language using the current time from the system prompt.",
      {"text": S, "when": S, "project": S}, ["text", "when"])
async def set_reminder(ctx, text, when, project=None):
    due = parse_when(when)
    p = find_project(ctx.con, project)
    rid = db.insert(ctx.con, "reminders", text=text, due_at=db.to_utc(due),
                    project_id=p["id"] if p else None, created_at=db.now())
    return {"id": rid, "due_local": fmt_local(db.to_utc(due))}


@tool("List open reminders.", {"include_done": {"type": "boolean"}})
async def list_reminders(ctx, include_done=False):
    rs = db.rows(ctx.con, "SELECT id, text, due_at, done FROM reminders "
                          + ("" if include_done else "WHERE done=0 ") + "ORDER BY due_at LIMIT 50")
    return {"reminders": [{**r, "due_local": fmt_local(r["due_at"])} for r in rs]}


@tool("Mark a reminder done.", {"reminder_id": I}, ["reminder_id"])
async def complete_reminder(ctx, reminder_id):
    if not db.one(ctx.con, "SELECT id FROM reminders WHERE id=?", reminder_id):
        raise ToolError(f"reminder {reminder_id} not found")
    db.update(ctx.con, "reminders", reminder_id, done=1)
    return {"done": reminder_id}


@tool("Record a decision made on a project and why.", {"project": S, "text": S, "why": S}, ["project", "text"])
async def log_decision(ctx, project, text, why=""):
    p = need_project(ctx.con, project)
    did = db.insert(ctx.con, "decisions", project_id=p["id"], text=text, why=why, source="chat", decided_at=db.now())
    return {"id": did, "project": p["name"]}


@tool("List a project's decision log.", {"project": S}, ["project"])
async def list_decisions(ctx, project):
    p = need_project(ctx.con, project)
    return {"decisions": db.rows(ctx.con, "SELECT text, why, source, decided_at FROM decisions "
                                          "WHERE project_id=? ORDER BY decided_at DESC", p["id"])}


@tool("Fetch a web page, summarize it and file it under a project.", {"url": S, "project": S}, ["url"])
async def capture_link(ctx, url, project=None):
    if not url.startswith(("http://", "https://")):
        raise ToolError("url must start with http:// or https://")
    body = b""
    async with httpx.AsyncClient(follow_redirects=True, timeout=10) as client:
        async with client.stream("GET", url, headers={"User-Agent": "Mozilla/5.0 personal-assistant"}) as r:
            r.raise_for_status()
            async for chunk in r.aiter_bytes():
                body += chunk
                if len(body) > 2_000_000:
                    break
            raw = body.decode(r.encoding or "utf-8", errors="replace")
    m = re.search(r"(?is)<title[^>]*>(.*?)</title>", raw)
    title = html.unescape(m.group(1)).strip()[:200] if m else url
    resp, _ = await ctx.llm.generate(
        f"Summarize this page in 3-5 bullet points for a developer's notes.\nURL: {url}\n\n{html_to_text(raw)[:20000]}")
    summary = (resp.text or "").strip()
    p = find_project(ctx.con, project)
    pid = p["id"] if p else None
    lid = db.insert(ctx.con, "links", url=url, title=title, summary=summary, project_id=pid, created_at=db.now())
    await memory.remember(ctx.con, ctx.llm, f"Saved link '{title}' ({url}): {summary[:400]}", "note", "link", pid)
    return {"id": lid, "title": title, "summary": summary, "project": p["name"] if p else None}


@tool("Get the latest daily or weekly recap.", {"period": {"type": "string", "enum": ["daily", "weekly"]}})
async def get_recap(ctx, period="daily"):
    r = db.one(ctx.con, "SELECT period_start, content FROM recaps WHERE period=? ORDER BY id DESC LIMIT 1", period)
    return {"recap": r} if r else {"recap": None, "note": "No recap yet; recaps are written nightly at 02:00."}


@tool("Rank what the user should work on now (due reminders, momentum, stalled projects).")
async def what_to_work_on(ctx):
    soon = db.to_utc(datetime.now(timezone.utc) + timedelta(days=2))
    due = db.rows(ctx.con, "SELECT id, text, due_at FROM reminders WHERE done=0 AND due_at <= ? ORDER BY due_at", soon)
    return {"projects": rank_projects(ctx.con),
            "due_reminders": [{**r, "due_local": fmt_local(r["due_at"])} for r in due]}


@tool("Build a paste-ready prompt to resume a project in a new Claude Code session.", {"project": S}, ["project"])
async def resume_context(ctx, project):
    return {"prompt": build_resume(ctx.con, need_project(ctx.con, project))}


@tool("Run Claude Code (full file/command access) inside a project's folder and return its answer. "
      "Use for anything about the project's code. For pure questions, tell Claude not to modify files. "
      "follow_up=true continues the previous Claude session for that project.",
      {"project": S, "prompt": S, "follow_up": {"type": "boolean"}}, ["project", "prompt"])
async def ask_claude_code(ctx, project, prompt, follow_up=False):
    p = need_project(ctx.con, project)
    if not p["path"]:
        raise ToolError(f"Project '{p['name']}' has no folder path. Ask the user for it, "
                        "then call update_project with path.")
    res = await ctx.runner.run(p, prompt, follow_up)
    return {"result": res["result"][:12000], "files_changed": res["files_changed"],
            "timed_out": res["timed_out"], "exit_code": res["exit_code"]}


@tool("Search the web (Google) for current info: libraries, papers, news, whether something already exists.",
      {"query": S}, ["query"])
async def web_search(ctx, query):
    resp, _ = await ctx.llm.search(query)
    gm = resp.candidates[0].grounding_metadata if resp.candidates else None
    chunks = (gm.grounding_chunks or []) if gm else []
    sources = [{"title": c.web.title, "url": c.web.uri} for c in chunks if c.web][:6]
    return {"answer": resp.text, "sources": sources}
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest -q test_core.py`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add tools.py test_core.py
git commit -m "feat: gemini tools for projects, ideas, reminders, links, search"
```

---

### Task 5: Claude Code runner

**Files:**
- Create: `claude_runner.py`
- Test: `test_core.py` (append)

**Interfaces:**
- Consumes: `db.insert`, `db.update`, `db.now`; project dict with `id`, `path`, `claude_session_id`.
- Produces: `ClaudeRunner(con, claude_bin="claude", timeout=600)`; `build_cmd(session_id=None) -> list[str]`; `async run(project, prompt, follow_up=False) -> {"result": str, "session_id": str|None, "files_changed": list[str], "exit_code": int, "timed_out": bool, "duration_s": float}`. Raises `ValueError` if folder missing. Saves `session_id` on the project and logs a `claude_runs` row.

- [ ] **Step 1: Append failing tests**

```python
import subprocess

from claude_runner import ClaudeRunner

FAKE_CLAUDE = """#!/usr/bin/env bash
PROMPT=$(cat)
case "$PROMPT" in
  sleep) exec sleep 5 ;;
  touch) echo hi > new.txt ;;
  empty) exit 0 ;;
esac
printf '{"type":"result","result":"cwd=%s args=%s","session_id":"sess-1"}' "$PWD" "$*"
"""


@pytest.fixture
def runner_env(con, tmp_path):
    fake = tmp_path / "claude"
    fake.write_text(FAKE_CLAUDE)
    fake.chmod(0o755)
    proj = tmp_path / "proj"
    proj.mkdir()
    subprocess.run(["git", "init", "-q", str(proj)], check=True)
    pid = db.insert(con, "projects", name="proj", path=str(proj), created_at=db.now())
    get = lambda: db.one(con, "SELECT * FROM projects WHERE id=?", pid)
    return ClaudeRunner(con, claude_bin=str(fake), timeout=2), get


def test_claude_run_in_project_dir(con, runner_env):
    runner, get = runner_env
    r = asyncio.run(runner.run(get(), "what providers?"))
    assert f"cwd={get()['path']}" in r["result"] and "bypassPermissions" in r["result"]
    assert "--resume" not in r["result"]
    assert get()["claude_session_id"] == "sess-1"
    assert db.one(con, "SELECT prompt FROM claude_runs")["prompt"] == "what providers?"
    r2 = asyncio.run(runner.run(get(), "and more?", follow_up=True))
    assert "--resume sess-1" in r2["result"]


def test_claude_files_changed_timeout_empty(con, runner_env):
    runner, get = runner_env
    assert asyncio.run(runner.run(get(), "touch"))["files_changed"] == ["?? new.txt"]
    assert asyncio.run(runner.run(get(), "sleep"))["timed_out"] is True
    assert "no output" in asyncio.run(runner.run(get(), "empty"))["result"]


def test_claude_missing_folder(con, runner_env):
    runner, _ = runner_env
    with pytest.raises(ValueError):
        asyncio.run(runner.run({"id": 1, "path": "/nope/x", "claude_session_id": None}, "hi"))
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest -q test_core.py -k claude`
Expected: FAIL — `No module named 'claude_runner'`

- [ ] **Step 3: Implement `claude_runner.py`**

```python
"""Run Claude Code headless inside a project folder (user's Pro plan does the heavy lifting)."""
import asyncio
import json
import os
import time
from pathlib import Path

import db


class ClaudeRunner:
    def __init__(self, con, claude_bin: str = "claude", timeout: int = 600):
        self.con, self.claude_bin, self.timeout = con, claude_bin, timeout
        self.lock = asyncio.Lock()  # one run at a time: avoids Pro rate limits and edit collisions

    def build_cmd(self, session_id: str | None = None) -> list[str]:
        # Full access by user decision. Prompt goes on stdin so a leading "-" can't become a flag.
        cmd = [self.claude_bin, "-p", "--output-format", "json", "--permission-mode", "bypassPermissions"]
        return cmd + ["--resume", session_id] if session_id else cmd

    async def _git_status(self, path: str) -> set[str] | None:
        proc = await asyncio.create_subprocess_exec(
            "git", "-C", path, "status", "--porcelain",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        out, _ = await proc.communicate()
        return set(out.decode().splitlines()) if proc.returncode == 0 else None

    async def run(self, project: dict, prompt: str, follow_up: bool = False) -> dict:
        path = project.get("path")
        if not path or not Path(path).is_dir():
            raise ValueError(f"project folder {path!r} does not exist")
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        async with self.lock:
            before = await self._git_status(path)
            session = project.get("claude_session_id") if follow_up else None
            start = time.monotonic()
            proc = await asyncio.create_subprocess_exec(
                *self.build_cmd(session), cwd=path, env=env,
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            timed_out = False
            try:
                out, err = await asyncio.wait_for(proc.communicate(prompt.encode()), self.timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                out, err, timed_out = b"", b"", True
            duration = round(time.monotonic() - start, 1)
            after = await self._git_status(path)
        result, session_id = self._parse(out.decode(errors="replace"), err.decode(errors="replace"))
        if timed_out:
            result = f"Claude Code timed out after {self.timeout}s and was stopped."
        # ponytail: files already dirty before the run and edited again won't show; diff content if that matters
        changed = sorted(after - before) if before is not None and after is not None else []
        db.insert(self.con, "claude_runs", project_id=project["id"], prompt=prompt, output=result,
                  session_id=session_id, files_changed=json.dumps(changed), exit_code=proc.returncode,
                  duration_s=duration, created_at=db.now())
        if session_id:
            db.update(self.con, "projects", project["id"], claude_session_id=session_id, last_activity_at=db.now())
        return {"result": result, "session_id": session_id, "files_changed": changed,
                "exit_code": proc.returncode, "timed_out": timed_out, "duration_s": duration}

    @staticmethod
    def _parse(out: str, err: str) -> tuple[str, str | None]:
        if not out.strip():
            return ("Claude Code returned no output (check `claude` login and network). "
                    + err.strip()[-2000:]).strip(), None
        try:
            d = json.loads(out)
            return str(d.get("result", "")), d.get("session_id")
        except json.JSONDecodeError:
            return out.strip()[-8000:], None
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest -q test_core.py`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add claude_runner.py test_core.py
git commit -m "feat: headless claude code runner"
```

---

### Task 6: Agent loop, greeting, extraction

**Files:**
- Create: `agent.py`
- Test: `test_core.py` (append)

**Interfaces:**
- Consumes: `tools.Ctx`, `tools.declarations()`, `tools.run_tool`, `tools.fmt_local`, `tools.find_project`, `memory.recall`, `memory.remember`, `llm.AllModelsExhausted`.
- Produces: `async agent.chat_turn(ctx, text)` → async generator of events `{"type": "tool", "name", "args"}`, `{"type": "tool_result", "name", "result"}`, `{"type": "answer", "text", "model", "memories", "tools"}`, `{"type": "error", "text"}`;
  `agent.start_turn(ctx, text) -> (asyncio.Queue, asyncio.Task)` (queue ends with `None`; task keeps running if nobody reads);
  `async agent.greeting(ctx) -> str`; `agent.spawn(coro) -> Task`.
- Messages saved to `messages` with `meta` JSON (`model`, `memories`, `tools`).

- [ ] **Step 1: Append failing tests**

```python
from google.genai import types as gtypes

import agent


def call_resp(name, args):
    part = gtypes.Part(function_call=gtypes.FunctionCall(name=name, args=args))
    return SimpleNamespace(text=None, function_calls=[part.function_call],
                           candidates=[SimpleNamespace(content=gtypes.Content(role="model", parts=[part]))])


async def collect(gen):
    return [e async for e in gen]


def new_conv(con):
    return db.insert(con, "conversations", created_at=db.now(), updated_at=db.now())


def test_chat_turn_runs_tool_then_answers(con):
    cid = new_conv(con)
    fl = FakeLLM(replies=[call_resp("save_idea", {"text": "GPU price tracker"}), "Saved your idea."])
    events = asyncio.run(collect(agent.chat_turn(tools.Ctx(con, fl, None, cid), "idea: GPU price tracker")))
    assert [e["type"] for e in events] == ["tool", "tool_result", "answer"]
    assert db.one(con, "SELECT text FROM ideas")["text"] == "GPU price tracker"
    msgs = db.rows(con, "SELECT role, content, meta FROM messages ORDER BY id")
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert json.loads(msgs[1]["meta"])["tools"][0]["name"] == "save_idea"
    assert db.one(con, "SELECT title FROM conversations")["title"] == "idea: GPU price tracker"


def test_chat_turn_all_models_exhausted_keeps_user_message(con):
    cid = new_conv(con)
    fl = FakeLLM(replies=[AllModelsExhausted(time.time() + 3600)])
    events = asyncio.run(collect(agent.chat_turn(tools.Ctx(con, fl, None, cid), "hello")))
    assert events[-1]["type"] == "error" and "out of quota" in events[-1]["text"]
    assert db.one(con, "SELECT content FROM messages WHERE role='user'")["content"] == "hello"


def test_start_turn_finishes_without_consumer(con):
    cid = new_conv(con)

    async def go():
        _, task = agent.start_turn(tools.Ctx(con, FakeLLM(replies=["done!"]), None, cid), "hi")
        await task

    asyncio.run(go())
    assert db.one(con, "SELECT content FROM messages WHERE role='assistant'")["content"] == "done!"


def test_greeting_uses_llm_or_falls_back(con):
    db.insert(con, "reminders", text="call bank", due_at=db.to_utc(datetime.now(timezone.utc) + timedelta(hours=1)),
              created_at=db.now())
    assert asyncio.run(agent.greeting(tools.Ctx(con, FakeLLM(replies=["Morning!"]), None))) == "Morning!"
    fallback = asyncio.run(agent.greeting(tools.Ctx(con, FakeLLM(replies=[RuntimeError("down")]), None)))
    assert "call bank" in fallback
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest -q test_core.py -k "chat_turn or start_turn or greeting"`
Expected: FAIL — `No module named 'agent'`

- [ ] **Step 3: Implement `agent.py`**

```python
"""Chat turn loop (Gemini + tools), detached turns, greeting, memory extraction, rolling summary."""
import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone

from google.genai import types

import config
import db
import memory
import tools
from llm import AllModelsExhausted

log = logging.getLogger("agent")
HISTORY = 20
MAX_ROUNDS = 6
_background: set[asyncio.Task] = set()

SYSTEM = """You are a proactive personal assistant and second brain for one person (the user), \
a developer running many AI/ML projects. You track their projects, ideas, decisions and reminders over months.

How you work:
- Act, don't just talk: use tools to save ideas, update projects, set reminders, log decisions and search \
memory. When intent is clear, do it and confirm briefly afterwards.
- Connect the dots: when something relates to an existing project, idea or memory, say so.
- Questions about the code inside a project -> ask_claude_code (runs Claude Code in that folder with full access; \
for pure questions tell it not to modify files). If a project has no folder, ask the user for the path.
- Current info, libraries, papers, "has someone built this?" -> web_search.
- Be concise and direct, use markdown, and push back when an idea has obvious problems.
- User timezone: {tz}. Now: {now}. For reminders convert natural language to an ISO local datetime.

About the user:
{profile}

Active projects:
{projects}

Due or overdue reminders:
{reminders}

Relevant memories:
{memories}

Earlier in this conversation (summary):
{summary}"""

EXTRACT = """From this exchange, extract durable facts worth remembering long-term: about the user, their \
preferences, projects, decisions or plans. Skip small talk, temporary details, and anything the assistant \
already saved with tools this turn. Return JSON: \
{{"memories": [{{"text": "...", "kind": "fact|preference|decision|note", "project": "project name or null"}}]}}

User: {user}
Assistant: {answer}"""

SUMMARY = """Update this running summary of a conversation with the older messages below. \
Keep it under 200 words and keep concrete facts.

Current summary: {summary}

Messages:
{transcript}"""

GREETING = """Write a short greeting (max 80 words) for the user opening their assistant. Local time: {now}. \
Mention only what matters from the data below and suggest one concrete next move. No filler.

{facts}"""


def spawn(coro) -> asyncio.Task:
    t = asyncio.create_task(coro)
    _background.add(t)
    t.add_done_callback(_background.discard)
    return t


def local_now() -> str:
    return datetime.now(config.TZ).strftime("%A %Y-%m-%d %H:%M")


def projects_block(con) -> str:
    now = datetime.now(timezone.utc)
    lines = []
    for p in db.rows(con, "SELECT name, status, next_step, last_activity_at, path FROM projects "
                          "WHERE status != 'done' ORDER BY last_activity_at DESC LIMIT 25"):
        idle = f"{(now - db.parse(p['last_activity_at'])).days}d idle" if p["last_activity_at"] else "no activity"
        folder = "has folder" if p["path"] else "no folder"
        lines.append(f"- {p['name']} [{p['status']}, {idle}, {folder}] next: {p['next_step'] or '-'}")
    return "\n".join(lines) or "(none yet)"


def reminders_block(con) -> str:
    soon = db.to_utc(datetime.now(timezone.utc) + timedelta(days=1))
    rs = db.rows(con, "SELECT id, text, due_at FROM reminders WHERE done=0 AND due_at <= ? ORDER BY due_at", soon)
    return "\n".join(f"- #{r['id']} {r['text']} (due {tools.fmt_local(r['due_at'])})" for r in rs) or "(none)"


async def build_system(ctx, user_text: str) -> tuple[str, list[dict]]:
    mems = await memory.recall(ctx.con, ctx.llm, user_text)
    conv = db.one(ctx.con, "SELECT summary FROM conversations WHERE id=?", ctx.conversation_id) or {}
    text = SYSTEM.format(
        tz=config.TZ.key, now=local_now(),
        profile=db.kv_get(ctx.con, "profile", "(not learned yet)"),
        projects=projects_block(ctx.con), reminders=reminders_block(ctx.con),
        memories="\n".join(f"- {m['text']}" for m in mems) or "(none)",
        summary=conv.get("summary") or "(none)")
    return text, mems


def history_contents(con, cid: int) -> list[types.Content]:
    msgs = db.rows(con, "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT ?",
                   cid, HISTORY)[::-1]
    contents = [types.Content(role="user" if m["role"] == "user" else "model", parts=[types.Part(text=m["content"])])
                for m in msgs]
    if contents and contents[0].role == "model":  # history may start with the greeting
        contents.insert(0, types.Content(role="user", parts=[types.Part(text="(opened the app)")]))
    return contents


def _short(result: dict) -> dict:
    s = json.dumps(result, default=str)
    return result if len(s) <= 4000 else {"truncated": s[:4000]}


async def chat_turn(ctx, text: str):
    con, cid = ctx.con, ctx.conversation_id
    db.insert(con, "messages", conversation_id=cid, role="user", content=text, created_at=db.now())
    con.execute("UPDATE conversations SET title=? WHERE id=? AND title='New chat'", (text[:60], cid))
    tool_log, model = [], None
    try:
        system, used = await build_system(ctx, text)
        contents = history_contents(con, cid)
        cfg = types.GenerateContentConfig(
            system_instruction=system, tools=tools.declarations(),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
        answer = "I hit the tool-call limit for this turn; ask me to continue."
        for _ in range(MAX_ROUNDS):
            resp, model = await ctx.llm.generate(contents, cfg)
            calls = resp.function_calls or []
            if not calls:
                answer = resp.text or "(no answer)"
                break
            contents.append(resp.candidates[0].content)  # keeps Gemini 3 thought signatures intact
            parts = []
            for call in calls:
                args = dict(call.args or {})
                yield {"type": "tool", "name": call.name, "args": args}
                result = await tools.run_tool(ctx, call.name, args)
                tool_log.append({"name": call.name, "args": args, "result": _short(result)})
                yield {"type": "tool_result", "name": call.name, "result": _short(result)}
                parts.append(types.Part.from_function_response(name=call.name, response=result))
            contents.append(types.Content(role="user", parts=parts))
    except AllModelsExhausted as e:
        until = datetime.fromtimestamp(e.until, config.TZ).strftime("%H:%M")
        yield {"type": "error", "text": f"All Gemini models are out of quota until {until} ({config.TZ.key})."}
        return
    except Exception as e:
        log.exception("chat turn failed")
        yield {"type": "error", "text": f"Gemini error: {e}"}
        return
    meta = {"model": model, "memories": used, "tools": tool_log}
    db.insert(con, "messages", conversation_id=cid, role="assistant", content=answer,
              meta=json.dumps(meta, default=str), created_at=db.now())
    db.update(con, "conversations", cid, updated_at=db.now())
    yield {"type": "answer", "text": answer, **meta}
    spawn(after_turn(ctx, text, answer))


def start_turn(ctx, text: str) -> tuple[asyncio.Queue, asyncio.Task]:
    """Run a turn detached from the HTTP request so a closed tab doesn't cancel it."""
    queue: asyncio.Queue = asyncio.Queue()

    async def run():
        try:
            async for ev in chat_turn(ctx, text):
                await queue.put(ev)
        finally:
            await queue.put(None)

    return queue, spawn(run())


async def after_turn(ctx, user_text: str, answer: str) -> None:
    try:
        resp, _ = await ctx.llm.generate(
            EXTRACT.format(user=user_text[:4000], answer=answer[:4000]),
            types.GenerateContentConfig(response_mime_type="application/json"), background=True)
        for m in json.loads(resp.text).get("memories", [])[:5]:
            p = tools.find_project(ctx.con, m.get("project"))
            await memory.remember(ctx.con, ctx.llm, m["text"], m.get("kind") or "note", "chat", p["id"] if p else None)
        await maybe_summarize(ctx)
    except Exception:
        log.warning("post-turn extraction skipped", exc_info=True)


async def maybe_summarize(ctx) -> None:
    cid = ctx.conversation_id
    keep = db.rows(ctx.con, "SELECT id FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT ?", cid, HISTORY)
    if len(keep) < HISTORY:
        return
    conv = db.one(ctx.con, "SELECT summary, summary_upto FROM conversations WHERE id=?", cid)
    old = db.rows(ctx.con, "SELECT id, role, content FROM messages WHERE conversation_id=? AND id > ? AND id < ? "
                           "ORDER BY id", cid, conv["summary_upto"], keep[-1]["id"])
    if len(old) < 10:
        return
    transcript = "\n".join(f"{m['role']}: {m['content'][:1000]}" for m in old)
    resp, _ = await ctx.llm.generate(SUMMARY.format(summary=conv["summary"] or "(none)", transcript=transcript),
                                     background=True)
    db.update(ctx.con, "conversations", cid, summary=resp.text.strip(), summary_upto=old[-1]["id"])


async def greeting(ctx) -> str:
    con = ctx.con
    now = datetime.now(timezone.utc)
    last_seen = db.kv_get(con, "last_seen")
    db.kv_set(con, "last_seen", db.now())
    stalled = db.rows(con, "SELECT name, last_activity_at FROM projects WHERE status IN ('active', 'stalled') "
                           "AND last_activity_at < ? ORDER BY last_activity_at LIMIT 5",
                      db.to_utc(now - timedelta(days=7)))
    updated = db.rows(con, "SELECT name, next_step FROM projects WHERE last_activity_at > ? "
                           "ORDER BY last_activity_at DESC LIMIT 5", last_seen) if last_seen else []
    recap = db.one(con, "SELECT content FROM recaps WHERE period='daily' ORDER BY id DESC LIMIT 1")
    facts = "\n".join([
        "Due within a day:", reminders_block(con), "",
        "Stalled projects (>7 days idle):",
        "\n".join(f"- {p['name']} ({(now - db.parse(p['last_activity_at'])).days}d)" for p in stalled) or "(none)", "",
        "Projects with new activity since last visit:",
        "\n".join(f"- {p['name']}: next {p['next_step'] or '-'}" for p in updated) or "(none)", "",
        "Latest daily recap:", recap["content"][:1500] if recap else "(none)"])
    try:
        resp, _ = await ctx.llm.generate(GREETING.format(now=local_now(), facts=facts))
        return (resp.text or "").strip() or facts
    except Exception:
        log.warning("greeting fell back to template", exc_info=True)
        return f"Hi! Gemini is unavailable right now; here's where things stand.\n\n{facts}"
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest -q test_core.py`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add agent.py test_core.py
git commit -m "feat: agent chat loop with tools, greeting and memory extraction"
```

---

### Task 7: Background jobs (sync, nightly, scheduler)

**Files:**
- Create: `jobs.py`
- Test: `test_core.py` (append)

**Interfaces:**
- Consumes: `db.*`, `memory.remember`, `memory.DUP`, `tools.embed_project`, `tools.STATUSES`, `llm.pack/unpack`, `llm.AllModelsExhausted`, `config.TZ/CLAUDE_HISTORY_DIR/SYNC_MAX_CALLS/ROOT`.
- Produces: `jobs.parse_transcript(path, max_chars=30000) -> {"cwd", "last_ts", "text"} | None`;
  `async jobs.sync_claude_history(con, llm, history_dir, max_calls) -> {"processed","skipped","failed","deferred"} | {"busy": True}`;
  `jobs.mark_stalled(con, now=None)`; `jobs.merge_duplicates(con) -> int`; `async jobs.backfill_embeddings(con, llm) -> int`;
  `async jobs.write_recap(con, llm, period, start: date, end: date) -> str`; `jobs.backup(con, backup_dir, today: date, keep=7)`;
  `async jobs.nightly(con, llm, backup_dir, today: date)`; `async jobs.tick(con, llm, now=None)`; `async jobs.scheduler(con, llm)`.

- [ ] **Step 1: Append failing tests**

```python
import jobs


def write_jsonl(path, lines, tail=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(l) for l in lines) + "\n" + tail)


def transcript_lines(cwd):
    base = {"cwd": str(cwd), "timestamp": "2026-09-10T06:16:00.000Z", "isSidechain": False}
    return [
        {**base, "type": "user", "message": {"role": "user", "content": "add key rotation"}},
        {**base, "type": "user", "message": {"role": "user", "content": "<command-name>/model</command-name>"}},
        {**base, "type": "user", "isMeta": True, "message": {"role": "user", "content": [{"type": "text", "text": "meta"}]}},
        {**base, "type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Read"}]}},
        {**base, "type": "user", "message": {"content": [{"type": "tool_result", "content": "file"}]}},
        {**base, "type": "assistant", "timestamp": "2026-09-10T07:00:00.000Z",
         "message": {"content": [{"type": "thinking"}, {"type": "text", "text": "Done: rotation added."}]}},
        {"type": "last-prompt", "sessionId": "x"},
    ]


def test_parse_transcript_skips_noise_and_partial_line(tmp_path):
    f = tmp_path / "h" / "p" / "s.jsonl"
    write_jsonl(f, transcript_lines(tmp_path), tail='{"type": "user", "mess')
    t = jobs.parse_transcript(f)
    assert t["text"] == "USER: add key rotation\n\nCLAUDE: Done: rotation added."
    assert t["cwd"] == str(tmp_path) and t["last_ts"] == "2026-09-10T07:00:00.000Z"


SYNC_JSON = json.dumps({"goal": "Manage API keys", "status": "active", "next_step": "add tests", "notes": "rotation done",
                        "decisions": [{"text": "Use SQLite", "why": "simple"}], "memories": []})


def test_sync_creates_project_and_skips_unchanged(con, tmp_path):
    proj = tmp_path / "key_manager"
    proj.mkdir()
    hist = tmp_path / "hist"
    write_jsonl(hist / "-x-key-manager" / "s1.jsonl", transcript_lines(proj))
    fl = FakeLLM(replies=[SYNC_JSON])
    assert asyncio.run(jobs.sync_claude_history(con, fl, hist, 30))["processed"] == 1
    p = db.one(con, "SELECT * FROM projects")
    assert (p["name"], p["path"], p["next_step"]) == ("key_manager", str(proj), "add tests")
    assert p["last_activity_at"] == "2026-09-10T07:00:00+00:00"
    assert db.one(con, "SELECT text FROM decisions")["text"] == "Use SQLite"
    assert asyncio.run(jobs.sync_claude_history(con, fl, hist, 30))["skipped"] == 1


def test_sync_bad_llm_json_marks_failed_and_synced(con, tmp_path):
    hist = tmp_path / "hist"
    write_jsonl(hist / "p" / "s.jsonl", transcript_lines(tmp_path))
    stats = asyncio.run(jobs.sync_claude_history(con, FakeLLM(replies=["garbage"]), hist, 30))
    assert stats["failed"] == 1
    assert asyncio.run(jobs.sync_claude_history(con, FakeLLM(), hist, 30))["skipped"] == 1


def test_sync_defers_on_quota_and_cap(con, tmp_path):
    hist = tmp_path / "hist"
    for i in range(3):
        write_jsonl(hist / "p" / f"s{i}.jsonl", transcript_lines(tmp_path))
    stats = asyncio.run(jobs.sync_claude_history(con, FakeLLM(replies=[SYNC_JSON]), hist, 1))
    assert stats["processed"] == 1 and stats["deferred"] == 2
    stats = asyncio.run(jobs.sync_claude_history(con, FakeLLM(replies=[AllModelsExhausted(0)]), hist, 30))
    assert stats["processed"] == 0 and stats["deferred"] >= 1


def test_mark_stalled_both_ways(con):
    now = datetime(2026, 9, 29, tzinfo=timezone.utc)
    old = db.insert(con, "projects", name="old", last_activity_at=db.to_utc(now - timedelta(days=8)), created_at=db.now())
    back = db.insert(con, "projects", name="back", status="stalled", last_activity_at=db.to_utc(now), created_at=db.now())
    jobs.mark_stalled(con, now)
    st = {r["name"]: r["status"] for r in db.rows(con, "SELECT name, status FROM projects")}
    assert st == {"old": "stalled", "back": "active"}


def test_merge_duplicates_keeps_newest(con):
    for i, v in enumerate([[1, 0], [0.99, 0.02], [0, 1]]):
        db.insert(con, "memories", text=f"m{i}", embedding=llm.pack(np.array(v, np.float32)),
                  created_at=db.now(), updated_at=f"2026-09-2{i}T00:00:00+00:00")
    assert jobs.merge_duplicates(con) == 1
    assert sorted(r["text"] for r in db.rows(con, "SELECT text FROM memories")) == ["m1", "m2"]


def test_quiet_recap_needs_no_llm(con):
    fl = FakeLLM()
    text = asyncio.run(jobs.write_recap(con, fl, "daily", date(2026, 9, 28), date(2026, 9, 28)))
    assert text.startswith("Quiet day") and fl.prompts == []


def test_backup_keeps_last_n(con, tmp_path):
    for d in range(1, 10):
        jobs.backup(con, tmp_path / "b", date(2026, 9, d), keep=7)
    assert len(list((tmp_path / "b").glob("assistant-*.db"))) == 7


def test_tick_runs_sync_every_3h_and_nightly_once(con, monkeypatch):
    calls = []

    async def fake_sync(*a, **k):
        calls.append("sync")
        return {"deferred": 0}

    async def fake_nightly(*a, **k):
        calls.append("nightly")

    monkeypatch.setattr(jobs, "sync_claude_history", fake_sync)
    monkeypatch.setattr(jobs, "nightly", fake_nightly)
    t = datetime(2026, 9, 29, 21, 0, tzinfo=timezone.utc)  # 02:30 IST, Sep 30
    asyncio.run(jobs.tick(con, None, t))
    asyncio.run(jobs.tick(con, None, t + timedelta(minutes=1)))
    assert calls == ["sync", "nightly"]
    asyncio.run(jobs.tick(con, None, t + timedelta(hours=3, minutes=1)))
    assert calls == ["sync", "nightly", "sync"]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest -q test_core.py -k "transcript or sync or stalled or merge or recap or backup or tick"`
Expected: FAIL — `No module named 'jobs'`

- [ ] **Step 3: Implement `jobs.py`**

```python
"""Background work: Claude Code history sync, nightly reflection/recaps/backup, and the scheduler loop."""
import asyncio
import hashlib
import json
import logging
import sqlite3
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path

import numpy as np
from google.genai import types

import config
import db
import memory
import tools
from llm import AllModelsExhausted, pack, unpack

log = logging.getLogger("jobs")
SYNC_LOCK = asyncio.Lock()
JSON_CFG = types.GenerateContentConfig(response_mime_type="application/json")

SYNC_PROMPT = """You maintain a project tracker. Below is a Claude Code session transcript for the project at {cwd}.
Current record: {record}
Update the record from the transcript. Return JSON:
{{"goal": "one sentence", "status": "active|stalled|paused|done", "next_step": "...",
  "notes": "current state, max 120 words",
  "decisions": [{{"text": "...", "why": "..."}}],
  "memories": [{{"text": "...", "kind": "fact|preference|decision|note"}}]}}
Only include durable decisions/memories not already in the record, max 5 each.

TRANSCRIPT:
{text}"""

PROFILE_PROMPT = """Rewrite this profile of the user (max 150 words: who they are, skills, what they work on, \
preferences, working style) using the memories below. Keep what is still true.

Current profile: {profile}

Memories:
{memories}"""

RECAP_PROMPT = """Write a {period} recap ({start} to {end}) for the user in markdown with sections \
"## Progress", "## Stalled", "## New ideas", "## Suggested next moves" (max 250 words). Be specific.

Data:
{data}"""


# ---------- Claude Code history sync ----------

def parse_transcript(path: Path, max_chars: int = 30000) -> dict | None:
    cwd = last_ts = None
    turns = []
    with path.open(errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue  # partial line of a session being written right now
            if d.get("type") not in ("user", "assistant") or d.get("isSidechain") or d.get("isMeta"):
                continue
            cwd = d.get("cwd") or cwd
            last_ts = d.get("timestamp") or last_ts
            content = (d.get("message") or {}).get("content")
            if isinstance(content, str):
                texts = [content]
            elif isinstance(content, list) and all(c.get("type") in ("text", "thinking") for c in content):
                texts = [c.get("text", "") for c in content if c.get("type") == "text"]
            else:
                continue  # tool_use / tool_result
            text = "\n".join(texts).strip()
            if not text or text.startswith("<"):  # slash-command and system wrappers
                continue
            turns.append(("USER: " if d["type"] == "user" else "CLAUDE: ") + text)
    if not turns or not cwd:
        return None
    return {"cwd": cwd, "last_ts": last_ts, "text": "\n\n".join(turns)[-max_chars:]}


async def apply_transcript(con, llm, t: dict) -> None:
    name = Path(t["cwd"]).name
    p = db.one(con, "SELECT * FROM projects WHERE path=?", t["cwd"]) or \
        db.one(con, "SELECT * FROM projects WHERE name=?", name)
    if not p:
        pid = db.insert(con, "projects", name=name, created_at=db.now(),
                        path=t["cwd"] if Path(t["cwd"]).is_dir() else None)
        p = db.one(con, "SELECT * FROM projects WHERE id=?", pid)
    record = {k: p[k] for k in ("goal", "status", "next_step", "notes")}
    resp, _ = await llm.generate(SYNC_PROMPT.format(cwd=t["cwd"], record=json.dumps(record), text=t["text"]),
                                 JSON_CFG, background=True)
    d = json.loads(resp.text)
    fields = {k: str(d[k]) for k in ("goal", "next_step", "notes") if d.get(k)}
    if d.get("status") in tools.STATUSES:
        fields["status"] = d["status"]
    last = db.to_utc(db.parse(t["last_ts"])) if t["last_ts"] else None
    if last and (not p["last_activity_at"] or last > p["last_activity_at"]):
        fields["last_activity_at"] = last
    db.update(con, "projects", p["id"], **fields)
    for dec in (d.get("decisions") or [])[:5]:
        db.insert(con, "decisions", project_id=p["id"], text=dec["text"], why=dec.get("why") or "",
                  source="claude-sync", decided_at=last or db.now())
    for m in (d.get("memories") or [])[:5]:
        await memory.remember(con, llm, m["text"], m.get("kind") or "note", "claude-sync", p["id"])
    await tools.embed_project(con, llm, p["id"])


async def sync_claude_history(con, llm, history_dir: Path, max_calls: int) -> dict:
    if SYNC_LOCK.locked():
        return {"busy": True}
    async with SYNC_LOCK:
        stats = {"processed": 0, "skipped": 0, "failed": 0, "deferred": 0}
        files = sorted(Path(history_dir).glob("*/*.jsonl"), key=lambda f: f.stat().st_mtime)  # oldest first
        quota_out = False
        for f in files:
            digest = hashlib.sha256(f.read_bytes()).hexdigest()
            st = db.one(con, "SELECT sha256 FROM sync_state WHERE file_path=?", str(f))
            if st and st["sha256"] == digest:
                stats["skipped"] += 1
                continue
            if quota_out or stats["processed"] + stats["failed"] >= max_calls:
                stats["deferred"] += 1
                continue
            try:
                t = parse_transcript(f)
                if t:
                    await apply_transcript(con, llm, t)
                stats["processed"] += 1
            except AllModelsExhausted:
                quota_out = True
                stats["deferred"] += 1
                continue  # not marked: retried next run
            except Exception:
                log.exception("sync failed for %s", f)
                stats["failed"] += 1  # marked below so a bad file isn't retried until it changes
            con.execute("INSERT INTO sync_state(file_path, sha256, synced_at) VALUES(?, ?, ?) "
                        "ON CONFLICT(file_path) DO UPDATE SET sha256=excluded.sha256, synced_at=excluded.synced_at",
                        (str(f), digest, db.now()))
        log.info("claude history sync: %s", stats)
        return stats


# ---------- nightly ----------

def mark_stalled(con, now: datetime | None = None) -> None:
    cutoff = db.to_utc((now or datetime.now(timezone.utc)) - timedelta(days=7))
    con.execute("UPDATE projects SET status='stalled' WHERE status='active' AND last_activity_at < ?", (cutoff,))
    con.execute("UPDATE projects SET status='active' WHERE status='stalled' AND last_activity_at >= ?", (cutoff,))


def merge_duplicates(con) -> int:
    # ponytail: O(n^2) similarity matrix, fine to ~5k memories; switch to per-row top_k beyond that
    ms = db.rows(con, "SELECT id, embedding FROM memories WHERE embedding IS NOT NULL ORDER BY updated_at DESC")
    if len(ms) < 2:
        return 0
    mat = np.stack([unpack(m["embedding"]) for m in ms])
    mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    sims = mat @ mat.T
    removed: set[int] = set()
    for i in range(len(ms)):
        if ms[i]["id"] in removed:
            continue
        for j in np.nonzero(sims[i, i + 1:] >= memory.DUP)[0] + i + 1:
            removed.add(ms[j]["id"])  # older duplicate of a newer memory
    con.executemany("DELETE FROM memories WHERE id=?", [(i,) for i in removed])
    return len(removed)


async def backfill_embeddings(con, llm) -> int:
    n = 0
    for table in ("memories", "ideas"):
        rows = db.rows(con, f"SELECT id, text FROM {table} WHERE embedding IS NULL LIMIT 100")
        if not rows:
            continue
        vecs = await llm.embed([r["text"] for r in rows])
        if vecs is None:
            return n
        for r, v in zip(rows, vecs):
            db.update(con, table, r["id"], embedding=pack(v))
            n += 1
    for p in db.rows(con, "SELECT id FROM projects WHERE embedding IS NULL LIMIT 100"):
        await tools.embed_project(con, llm, p["id"])
    return n


async def rewrite_profile(con, llm) -> None:
    mems = db.rows(con, "SELECT text FROM memories WHERE kind IN ('preference', 'fact') "
                        "ORDER BY updated_at DESC LIMIT 60")
    if not mems:
        return
    resp, _ = await llm.generate(PROFILE_PROMPT.format(
        profile=db.kv_get(con, "profile", "(empty)"), memories="\n".join(f"- {m['text']}" for m in mems)),
        background=True)
    db.kv_set(con, "profile", resp.text.strip())


async def write_recap(con, llm, period: str, start: date, end: date) -> str:
    s = db.to_utc(datetime.combine(start, dtime.min, config.TZ))
    e = db.to_utc(datetime.combine(end + timedelta(days=1), dtime.min, config.TZ))
    msgs = db.rows(con, "SELECT content FROM messages WHERE role='user' AND created_at >= ? AND created_at < ? "
                        "ORDER BY id LIMIT 80", s, e)
    touched = db.rows(con, "SELECT name, status, next_step, notes FROM projects "
                           "WHERE last_activity_at >= ? AND last_activity_at < ?", s, e)
    ideas = db.rows(con, "SELECT text FROM ideas WHERE created_at >= ? AND created_at < ?", s, e)
    decisions = db.rows(con, "SELECT d.text, p.name AS project FROM decisions d JOIN projects p ON p.id=d.project_id "
                             "WHERE d.decided_at >= ? AND d.decided_at < ?", s, e)
    if not (msgs or touched or ideas or decisions):
        content = f"Quiet {'day' if period == 'daily' else 'week'}: no chats, project activity or new ideas."
    else:
        stalled = [r["name"] for r in db.rows(con, "SELECT name FROM projects WHERE status='stalled'")]
        data = json.dumps({"user_messages": [m["content"][:300] for m in msgs], "projects_worked_on": touched,
                           "new_ideas": [i["text"] for i in ideas], "decisions": decisions, "stalled": stalled},
                          ensure_ascii=False)[:30000]
        resp, _ = await llm.generate(RECAP_PROMPT.format(period=period, start=start, end=end, data=data),
                                     background=True)
        content = resp.text.strip()
    db.insert(con, "recaps", period=period, period_start=start.isoformat(), content=content, created_at=db.now())
    return content


def backup(con, backup_dir: Path, today: date, keep: int = 7) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    dst = sqlite3.connect(backup_dir / f"assistant-{today.isoformat()}.db")
    con.backup(dst)
    dst.close()
    for old in sorted(backup_dir.glob("assistant-*.db"))[:-keep]:
        old.unlink()


async def nightly(con, llm, backup_dir: Path, today: date) -> None:
    yesterday = today - timedelta(days=1)
    steps = [
        ("backfill", backfill_embeddings(con, llm)),
        ("profile", rewrite_profile(con, llm)),
        ("daily recap", write_recap(con, llm, "daily", yesterday, yesterday)),
    ]
    if yesterday.weekday() == 6:  # Sunday just ended
        steps.append(("weekly recap", write_recap(con, llm, "weekly", yesterday - timedelta(days=6), yesterday)))
    for name, fn in [("merge", merge_duplicates), ("stalled", mark_stalled)]:
        try:
            fn(con)
        except Exception:
            log.exception("nightly %s failed", name)
    for name, coro in steps:
        try:
            await coro
        except Exception:
            log.exception("nightly %s failed", name)
    try:
        backup(con, backup_dir, today)
    except Exception:
        log.exception("nightly backup failed")


# ---------- scheduler ----------

async def tick(con, llm, now: datetime | None = None) -> None:
    now = now or datetime.now(timezone.utc)
    if db.to_utc(now) >= (db.kv_get(con, "next_sync") or ""):
        stats = await sync_claude_history(con, llm, config.CLAUDE_HISTORY_DIR, config.SYNC_MAX_CALLS)
        delay = timedelta(minutes=10) if stats.get("deferred") else timedelta(hours=3)
        db.kv_set(con, "next_sync", db.to_utc(now + delay))
    local = now.astimezone(config.TZ)
    if local.hour >= 2 and db.kv_get(con, "last_nightly") != local.date().isoformat():
        db.kv_set(con, "last_nightly", local.date().isoformat())  # set first: a crash must not loop every minute
        await nightly(con, llm, config.ROOT / "backups", local.date())


async def scheduler(con, llm) -> None:
    while True:
        try:
            await tick(con, llm)
        except Exception:
            log.exception("scheduler tick failed")
        await asyncio.sleep(60)
```

Note: in `nightly`, the coroutines in `steps` are created before running; all are awaited in the loop, so none leak.

- [ ] **Step 4: Run tests**

Run: `uv run pytest -q test_core.py`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add jobs.py test_core.py
git commit -m "feat: claude history sync, nightly reflection and scheduler"
```

---

### Task 8: FastAPI app, auth, API routes

**Files:**
- Create: `main.py`, `static/login.html`
- Test: `test_core.py` (append)

**Interfaces:**
- Consumes: everything above. `agent.start_turn`, `agent.greeting`, `tools.run_tool/build_resume/rank_projects/fmt_local`, `memory.recall/remember/related`, `jobs.scheduler/sync_claude_history`, `llm.unpack/pack`.
- Produces: `main.create_app(con, llm, runner, start_jobs=True) -> FastAPI`; `main.main()`.
- HTTP API (JSON; all but `/login` need cookie; errors are `{"detail": msg}`):
  - `GET /login` page, `POST /login {"password"}`, `POST /logout`
  - `GET /` UI
  - `GET/POST /api/conversations`, `GET /api/conversations/{id}/messages`
  - `POST /api/chat {"conversation_id", "text"}` → SSE `data: {event}\n\n`
  - `GET/POST /api/projects`, `GET/PATCH /api/projects/{id}`, `GET /api/projects/{id}/resume`
  - `GET /api/ideas?status=`, `POST /api/ideas`, `PATCH /api/ideas/{id}`, `POST /api/ideas/{id}/promote`, `GET /api/ideas/{id}/related`
  - `GET /api/memories?q=`, `POST /api/memories`, `PATCH/DELETE /api/memories/{id}`, `GET/PUT /api/profile`
  - `GET /api/today`, `POST /api/reminders`, `PATCH /api/reminders/{id}`
  - `GET /api/status`, `POST /api/sync`

- [ ] **Step 1: Append failing tests**

```python
@pytest.fixture
def client(con, monkeypatch):
    from fastapi.testclient import TestClient

    import main
    monkeypatch.setattr(config, "APP_PASSWORD", "pw")
    monkeypatch.setattr(config, "SESSION_SECRET", "s" * 32)
    return TestClient(main.create_app(con, FakeLLM(replies=["Hello there"]), None, start_jobs=False))


def test_api_requires_login(client):
    assert client.get("/api/projects").status_code == 401
    assert client.get("/", follow_redirects=False).status_code in (302, 307)
    assert client.get("/login").status_code == 200


def test_login_rate_limit(client):
    for _ in range(5):
        assert client.post("/login", json={"password": "nope"}).status_code == 401
    assert client.post("/login", json={"password": "pw"}).status_code == 429


def test_login_then_chat_streams_answer(client, con):
    assert client.post("/login", json={"password": "pw"}).status_code == 200
    cid = db.insert(con, "conversations", created_at=db.now(), updated_at=db.now())
    r = client.post("/api/chat", json={"conversation_id": cid, "text": "hi"})
    assert '"type": "answer"' in r.text and "Hello there" in r.text
    assert client.get("/api/projects").json() == []
    assert client.post("/api/projects", json={"name": "x", "path": "/definitely/missing"}).status_code == 400
    assert client.get("/api/today").json()["work_on"] == []
    a = db.insert(con, "projects", name="key-manager", created_at=db.now())
    db.insert(con, "projects", name="key_manager", created_at=db.now())
    assert client.patch(f"/api/projects/{a}", json={"next_step": "ship"}).status_code == 200
    assert client.patch(f"/api/projects/{a}", json={"status": "bogus"}).status_code == 400
    assert db.one(con, "SELECT next_step FROM projects WHERE id=?", a)["next_step"] == "ship"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest -q test_core.py -k "login or chat_streams"`
Expected: FAIL — `No module named 'main'`

- [ ] **Step 3: Create `static/login.html`**

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Assistant · Log in</title>
<style>
  body{margin:0;min-height:100dvh;display:grid;place-items:center;background:#0f1115;color:#e6e8ee;
       font:16px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
  form{width:min(320px,90vw);display:grid;gap:12px}
  h1{font-size:20px;margin:0 0 8px}
  input{padding:12px;border-radius:8px;border:1px solid #262b36;background:#171a21;color:inherit;font:inherit}
  button{padding:12px;border:0;border-radius:8px;background:#7aa2ff;color:#0b0d12;font:inherit;font-weight:600}
  #e{color:#ff7a7a;min-height:1.5em;margin:0}
</style>
</head>
<body>
<form id="f">
  <h1>🧠 Assistant</h1>
  <input type="password" id="p" placeholder="Password" autocomplete="current-password" autofocus>
  <button>Log in</button>
  <p id="e"></p>
</form>
<script>
document.getElementById('f').onsubmit = async ev => {
  ev.preventDefault();
  const r = await fetch('/login', {method: 'POST', headers: {'Content-Type': 'application/json'},
                                   body: JSON.stringify({password: document.getElementById('p').value})});
  if (r.ok) location = '/';
  else document.getElementById('e').textContent = (await r.json().catch(() => ({}))).detail || 'Login failed';
};
</script>
</body>
</html>
```

- [ ] **Step 4: Implement `main.py`**

```python
"""FastAPI app: password auth, JSON API, SSE chat, background scheduler. Run: uv run main.py"""
import asyncio
import hmac
import json
import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from google import genai
from google.genai import types
from itsdangerous import BadSignature, URLSafeTimedSerializer
from pydantic import BaseModel

import agent
import config
import db
import jobs
import memory
import tools
from claude_runner import ClaudeRunner
from llm import LLM, pack, unpack

log = logging.getLogger("main")
STATIC = config.ROOT / "static"
COOKIE = "session"
MAX_AGE = 30 * 86400
PROJECT_EDITABLE = {"goal", "status", "next_step", "notes", "path", "main_files"}
IDEA_STATUSES = {"inbox", "promoted", "dropped"}


class LoginIn(BaseModel):
    password: str


class ChatIn(BaseModel):
    conversation_id: int
    text: str


def pick(body: dict, allowed: set) -> dict:
    return {k: v for k, v in body.items() if k in allowed and v is not None}


def create_app(con, llm, runner, start_jobs: bool = True) -> FastAPI:
    signer = URLSafeTimedSerializer(config.SESSION_SECRET, salt="login")
    failures: dict[str, deque] = defaultdict(deque)

    @asynccontextmanager
    async def lifespan(app):
        job = asyncio.create_task(jobs.scheduler(con, llm)) if start_jobs else None
        yield
        if job:
            job.cancel()

    app = FastAPI(lifespan=lifespan)

    def ctx(cid=None) -> tools.Ctx:
        return tools.Ctx(con, llm, runner, cid)

    async def run_tool(name: str, args: dict) -> dict:
        r = await tools.run_tool(ctx(), name, args)
        if "error" in r:
            raise HTTPException(400, r["error"])
        return r

    def get_or_404(table: str, id_: int) -> dict:
        r = db.one(con, f"SELECT * FROM {table} WHERE id=?", id_)
        if not r:
            raise HTTPException(404, "not found")
        return r

    def logged_in(request: Request) -> bool:
        token = request.cookies.get(COOKIE)
        if not token:
            return False
        try:
            signer.loads(token, max_age=MAX_AGE)
            return True
        except BadSignature:
            return False

    @app.middleware("http")
    async def auth(request: Request, call_next):
        if request.url.path == "/login" or logged_in(request):
            return await call_next(request)
        if request.url.path.startswith("/api/"):
            return JSONResponse({"detail": "login required"}, status_code=401)
        return RedirectResponse("/login")

    # ---------- auth ----------

    @app.get("/login")
    async def login_page():
        return FileResponse(STATIC / "login.html")

    @app.post("/login")
    async def login(body: LoginIn, request: Request):
        ip = request.client.host if request.client else "?"
        q, now = failures[ip], time.time()
        while q and q[0] < now - 900:
            q.popleft()
        if len(q) >= 5:
            raise HTTPException(429, "Too many attempts. Try again in 15 minutes.")
        if not hmac.compare_digest(body.password.encode(), config.APP_PASSWORD.encode()):
            q.append(now)
            raise HTTPException(401, "Wrong password")
        q.clear()
        resp = JSONResponse({"ok": True})
        resp.set_cookie(COOKIE, signer.dumps("me"), max_age=MAX_AGE, httponly=True, samesite="lax",
                        secure=request.url.scheme == "https")
        return resp

    @app.post("/logout")
    async def logout():
        resp = JSONResponse({"ok": True})
        resp.delete_cookie(COOKIE)
        return resp

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    # ---------- chat ----------

    @app.get("/api/conversations")
    async def conversations():
        return db.rows(con, "SELECT id, title, updated_at FROM conversations ORDER BY updated_at DESC LIMIT 50")

    @app.post("/api/conversations")
    async def new_conversation():
        cid = db.insert(con, "conversations", created_at=db.now(), updated_at=db.now())
        text = await agent.greeting(ctx(cid))
        db.insert(con, "messages", conversation_id=cid, role="assistant", content=text, created_at=db.now())
        return {"id": cid, "greeting": text}

    @app.get("/api/conversations/{cid}/messages")
    async def messages(cid: int):
        ms = db.rows(con, "SELECT id, role, content, meta, created_at FROM messages "
                          "WHERE conversation_id=? ORDER BY id", cid)
        for m in ms:
            m["meta"] = json.loads(m["meta"] or "{}")
        return ms

    @app.post("/api/chat")
    async def chat(body: ChatIn):
        text = body.text.strip()
        if not text:
            raise HTTPException(400, "empty message")
        get_or_404("conversations", body.conversation_id)
        queue, _ = agent.start_turn(ctx(body.conversation_id), text)

        async def stream():
            while (ev := await queue.get()) is not None:
                yield f"data: {json.dumps(ev, default=str)}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    # ---------- projects ----------

    @app.get("/api/projects")
    async def projects():
        return [db.public(p) for p in db.rows(
            con, "SELECT * FROM projects ORDER BY status='done', last_activity_at DESC")]

    @app.post("/api/projects")
    async def create_project(body: dict):
        return await run_tool("create_project", pick(body, PROJECT_EDITABLE | {"name"}))

    @app.get("/api/projects/{pid}")
    async def project(pid: int):
        p = db.public(get_or_404("projects", pid))
        p["decisions"] = db.rows(con, "SELECT * FROM decisions WHERE project_id=? ORDER BY decided_at DESC", pid)
        p["links"] = db.rows(con, "SELECT * FROM links WHERE project_id=? ORDER BY created_at DESC", pid)
        p["memories"] = db.rows(con, "SELECT id, text, kind FROM memories WHERE project_id=? "
                                     "ORDER BY updated_at DESC", pid)
        p["runs"] = db.rows(con, "SELECT id, prompt, output, files_changed, exit_code, duration_s, created_at "
                                 "FROM claude_runs WHERE project_id=? ORDER BY id DESC LIMIT 20", pid)
        return p

    @app.patch("/api/projects/{pid}")
    async def edit_project(pid: int, body: dict):
        get_or_404("projects", pid)  # by id: names like key-manager/key_manager would be ambiguous
        fields = pick(body, PROJECT_EDITABLE)
        try:
            tools.check_fields(fields)
        except tools.ToolError as e:
            raise HTTPException(400, str(e))
        db.update(con, "projects", pid, last_activity_at=db.now(), **fields)
        if "goal" in fields:
            await tools.embed_project(con, llm, pid)
        return {"ok": True}

    @app.get("/api/projects/{pid}/resume")
    async def resume(pid: int):
        return {"text": tools.build_resume(con, get_or_404("projects", pid))}

    # ---------- ideas ----------

    @app.get("/api/ideas")
    async def ideas(status: str = "inbox"):
        return [db.public(i) for i in db.rows(con, "SELECT * FROM ideas WHERE status=? "
                                                   "ORDER BY created_at DESC", status)]

    @app.post("/api/ideas")
    async def add_idea(body: dict):
        return await run_tool("save_idea", pick(body, {"text", "tags"}))

    @app.patch("/api/ideas/{iid}")
    async def edit_idea(iid: int, body: dict):
        get_or_404("ideas", iid)
        fields = pick(body, {"text", "tags", "status"})
        if fields.get("status") not in (None, *IDEA_STATUSES):
            raise HTTPException(400, f"status must be one of {sorted(IDEA_STATUSES)}")
        db.update(con, "ideas", iid, **fields)
        return {"ok": True}

    @app.post("/api/ideas/{iid}/promote")
    async def promote(iid: int):
        return await run_tool("promote_idea", {"idea_id": iid})

    @app.get("/api/ideas/{iid}/related")
    async def idea_related(iid: int):
        idea = get_or_404("ideas", iid)
        return memory.related(con, unpack(idea["embedding"]), exclude_idea=iid) if idea["embedding"] else []

    # ---------- memory ----------

    @app.get("/api/memories")
    async def memories(q: str = ""):
        if q.strip():
            return await memory.recall(con, llm, q, k=30)
        return db.rows(con, "SELECT id, text, kind, source, project_id, updated_at FROM memories "
                            "ORDER BY updated_at DESC LIMIT 200")

    @app.post("/api/memories")
    async def add_memory(body: dict):
        text = str(body.get("text", "")).strip()
        if not text:
            raise HTTPException(400, "empty memory")
        return await memory.remember(con, llm, text, body.get("kind") or "note", "manual")

    @app.patch("/api/memories/{mid}")
    async def edit_memory(mid: int, body: dict):
        get_or_404("memories", mid)
        fields = pick(body, {"text", "kind"})
        if "text" in fields:
            vecs = await llm.embed([fields["text"]])
            fields["embedding"] = pack(vecs[0]) if vecs else None
        db.update(con, "memories", mid, updated_at=db.now(), **fields)
        return {"ok": True}

    @app.delete("/api/memories/{mid}")
    async def delete_memory(mid: int):
        con.execute("DELETE FROM memories WHERE id=?", (mid,))
        return {"ok": True}

    @app.get("/api/profile")
    async def get_profile():
        return {"content": db.kv_get(con, "profile", "")}

    @app.put("/api/profile")
    async def put_profile(body: dict):
        db.kv_set(con, "profile", str(body.get("content", "")))
        return {"ok": True}

    # ---------- today / reminders ----------

    @app.get("/api/today")
    async def today():
        return {
            "reminders": db.rows(con, "SELECT r.*, p.name AS project FROM reminders r "
                                      "LEFT JOIN projects p ON p.id=r.project_id "
                                      "WHERE r.done=0 ORDER BY r.due_at LIMIT 50"),
            "work_on": tools.rank_projects(con),
            "recaps": db.rows(con, "SELECT * FROM recaps ORDER BY id DESC LIMIT 2"),
            "next_sync": db.kv_get(con, "next_sync"),
        }

    @app.post("/api/reminders")
    async def add_reminder(body: dict):
        return await run_tool("set_reminder", pick(body, {"text", "when", "project"}))

    @app.patch("/api/reminders/{rid}")
    async def edit_reminder(rid: int, body: dict):
        get_or_404("reminders", rid)
        db.update(con, "reminders", rid, done=1 if body.get("done") else 0)
        return {"ok": True}

    # ---------- status / sync ----------

    @app.get("/api/status")
    async def status():
        now = time.time()
        return {"model": llm.last_model, "timezone": config.TZ.key,
                "exhausted": {m: llm.exhausted_until(m) for m in llm.models if llm.exhausted_until(m) > now},
                "dead": sorted(llm.dead)}

    @app.post("/api/sync")
    async def sync_now():
        return await jobs.sync_claude_history(con, llm, config.CLAUDE_HISTORY_DIR, config.SYNC_MAX_CALLS)

    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(config.ROOT / "assistant.log")])
    missing = [k for k in ("GEMINI_API_KEY", "APP_PASSWORD", "SESSION_SECRET") if not getattr(config, k)]
    if missing:
        raise SystemExit(f"Missing in .env: {', '.join(missing)}")
    con = db.connect(config.DB_PATH)
    client = genai.Client(api_key=config.GEMINI_API_KEY, http_options=types.HttpOptions(
        retry_options=types.HttpRetryOptions(attempts=1)))
    llm = LLM(client, con, config.MODELS, config.EMBED_MODEL, config.SEARCH_MODELS, config.RPM_LIMIT)
    runner = ClaudeRunner(con, timeout=config.CLAUDE_TIMEOUT)
    uvicorn.run(create_app(con, llm, runner), host="0.0.0.0", port=config.PORT)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest -q test_core.py`
Expected: all pass (the `/` redirect test needs `static/index.html` only after login, so it passes now)

- [ ] **Step 6: Commit**

```bash
git add main.py static/login.html test_core.py
git commit -m "feat: fastapi app with auth, api routes and sse chat"
```

---

### Task 9: Web UI (mobile-first)

**Files:**
- Create: `static/index.html`

**Interfaces:**
- Consumes: HTTP API from Task 8 exactly as listed there. SSE events from Task 6.

- [ ] **Step 1: Create `static/index.html`**

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Assistant</title>
<style>
:root{--bg:#0f1115;--panel:#171a21;--line:#262b36;--text:#e6e8ee;--muted:#8b93a7;--acc:#7aa2ff;--warn:#ff7a7a;--ok:#5fd4a0;--amber:#f5c26b}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
body{display:grid;height:100dvh;grid-template-rows:auto 1fr auto;grid-template-areas:"h" "m" "n"}
header{grid-area:h;display:flex;align-items:center;gap:8px;padding:10px 16px;border-bottom:1px solid var(--line)}
header h1{font-size:16px;margin:0;flex:1}
main{grid-area:m;display:flex;min-height:0;overflow:hidden}
nav{grid-area:n;display:flex;background:var(--panel);border-top:1px solid var(--line)}
nav button{flex:1;position:relative;padding:10px 4px calc(10px + env(safe-area-inset-bottom));background:none;border:0;color:var(--muted);font:inherit;font-size:12px;cursor:pointer}
nav button.on{color:var(--acc)}
.badge{position:absolute;top:4px;margin-left:2px;background:var(--warn);color:#fff;border-radius:999px;font-size:10px;padding:0 5px}
.badge:empty{display:none}
section{display:none;flex:1;flex-direction:column;min-width:0;min-height:0;overflow:auto;padding:16px;max-width:900px;margin:0 auto}
section.on{display:flex}
#chat{overflow:hidden;padding-bottom:0}
#msgs{flex:1;overflow:auto;display:flex;flex-direction:column;gap:10px;padding-bottom:10px}
.msg{max-width:88%;padding:10px 12px;border-radius:12px;background:var(--panel);border:1px solid var(--line);overflow-wrap:anywhere}
.msg.user{align-self:flex-end;background:#1f2a44;border-color:#2c3b61}
.msg pre{background:#0b0d12;padding:8px;border-radius:8px;overflow:auto}
.msg code{font-family:ui-monospace,Menlo,monospace;font-size:13px}
.msg h4{margin:8px 0 4px}.msg li{margin-left:18px}
.tool{font:12px ui-monospace,Menlo,monospace;color:var(--muted);align-self:flex-start;max-width:100%}
.tool pre{white-space:pre-wrap;max-height:300px;overflow:auto}
.meta{margin-top:6px;display:flex;gap:6px;flex-wrap:wrap}
.chip{font-size:12px;color:var(--muted);border:1px solid var(--line);border-radius:999px;padding:2px 8px;background:none;cursor:pointer}
form.send{display:flex;gap:8px;padding:10px 0 12px;border-top:1px solid var(--line)}
input,textarea,select{width:100%;background:var(--bg);color:var(--text);border:1px solid var(--line);border-radius:8px;padding:9px 10px;font:inherit;font-size:16px}
form.send textarea{resize:none;height:44px;max-height:160px}
.btn{background:var(--acc);color:#0b0d12;border:0;border-radius:8px;padding:9px 14px;font:inherit;font-weight:600;cursor:pointer;white-space:nowrap}
.ghost{background:none;color:var(--text);border:1px solid var(--line);border-radius:8px;padding:8px 12px;font:inherit;cursor:pointer;white-space:nowrap}
.btn:disabled,.ghost:disabled{opacity:.5}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px;margin-bottom:10px}
.click{cursor:pointer}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.row>*:first-child{flex:1}
.muted{color:var(--muted)}.small{font-size:12px}
.st{font-size:11px;border-radius:999px;padding:1px 8px;border:1px solid var(--line)}
.st.active{color:var(--ok)}.st.stalled{color:var(--amber)}.st.done,.st.paused{color:var(--muted)}
.due{border-color:var(--warn)}
h2{font-size:15px;margin:18px 0 8px}
label{display:block;font-size:12px;color:var(--muted);margin:10px 0 4px}
.sheet{position:fixed;inset:0;background:var(--bg);overflow:auto;padding:16px;display:none;z-index:10}
.sheet.on{display:block}.sheet>div{max-width:900px;margin:0 auto}
#toast{position:fixed;left:50%;bottom:90px;transform:translateX(-50%);color:#0b0d12;padding:8px 14px;border-radius:8px;display:none;z-index:20;max-width:90vw}
@media(min-width:800px){
  body{grid-template-columns:170px 1fr;grid-template-rows:auto 1fr;grid-template-areas:"h h" "n m"}
  nav{flex-direction:column;border-top:0;border-right:1px solid var(--line)}
  nav button{flex:none;text-align:left;padding:12px 16px;font-size:14px}
  .badge{position:static}
}
</style>
</head>
<body>
<header>
  <h1>🧠 Assistant</h1>
  <button class="chip" id="model" onclick="refreshStatus()">…</button>
  <button class="chip" onclick="logout()">Log out</button>
</header>
<main>
  <section id="chat">
    <div class="row" style="margin-bottom:8px">
      <select id="convs" onchange="openConv(+this.value)"></select>
      <button class="ghost" onclick="newConv()">＋ New</button>
    </div>
    <div id="msgs"></div>
    <form class="send" onsubmit="send(event)">
      <textarea id="input" placeholder="Message, idea, or ask about a project…"
        onkeydown="if(event.key==='Enter'&&!event.shiftKey&&!matchMedia('(pointer:coarse)').matches){send(event)}"></textarea>
      <button class="btn" id="sendbtn">Send</button>
    </form>
  </section>

  <section id="projects">
    <div class="row"><h2>Projects</h2><button class="ghost" onclick="newProject()">＋ Project</button></div>
    <div id="plist"></div>
  </section>

  <section id="ideas">
    <form class="row" onsubmit="addIdea(event)" style="margin-bottom:12px">
      <input id="ideatext" placeholder="Quick idea…"><button class="btn">Add</button>
    </form>
    <div class="row" style="margin-bottom:8px">
      <h2>Ideas</h2>
      <select id="istatus" onchange="loadIdeas()" style="width:auto"><option>inbox</option><option>promoted</option><option>dropped</option></select>
    </div>
    <div id="ilist"></div>
  </section>

  <section id="memory">
    <h2>Profile</h2>
    <textarea id="profile" rows="5" placeholder="What the assistant knows about you (rewritten nightly, editable)"></textarea>
    <div class="row" style="margin-top:8px"><span></span><button class="ghost" onclick="saveProfile()">Save profile</button></div>
    <h2>Memories</h2>
    <form class="row" onsubmit="addMemory(event)"><input id="memnew" placeholder="Add a memory…"><button class="btn">Add</button></form>
    <input id="memq" placeholder="Search by meaning…" style="margin:10px 0"
      oninput="clearTimeout(this.t);this.t=setTimeout(loadMemory,400)">
    <div id="mlist"></div>
  </section>

  <section id="today">
    <h2>Reminders</h2>
    <form class="row" onsubmit="addReminder(event)">
      <input id="remtext" placeholder="Remind me to…">
      <input id="remwhen" type="datetime-local" style="width:auto">
      <button class="btn">Add</button>
    </form>
    <div id="rlist" style="margin-top:10px"></div>
    <h2>What to work on now</h2><div id="work"></div>
    <h2>Recaps</h2><div id="recaps"></div>
    <h2>Claude history sync</h2>
    <div class="card row"><span class="muted small" id="syncinfo"></span><button class="ghost" onclick="syncNow(this)">Sync now</button></div>
  </section>
</main>
<nav>
  <button data-v="chat" onclick="show('chat')">💬 Chat</button>
  <button data-v="projects" onclick="show('projects')">📁 Projects</button>
  <button data-v="ideas" onclick="show('ideas')">💡 Ideas</button>
  <button data-v="memory" onclick="show('memory')">🧠 Memory</button>
  <button data-v="today" onclick="show('today')">📅 Today<span class="badge" id="duebadge"></span></button>
</nav>
<div class="sheet" id="pdetail"><div id="pbody"></div></div>
<div id="toast"></div>

<script>
const $ = s => document.querySelector(s);
let TZ = 'Asia/Kolkata', conv = null, busy = false;

function toast(msg, ok = false) {
  const t = $('#toast');
  t.textContent = msg; t.style.background = ok ? 'var(--ok)' : 'var(--warn)'; t.style.display = 'block';
  clearTimeout(t.h); t.h = setTimeout(() => t.style.display = 'none', 4000);
}

async function api(path, opts = {}) {
  const init = {method: opts.method || 'GET', headers: {'Content-Type': 'application/json'}};
  if (opts.body !== undefined) init.body = JSON.stringify(opts.body);
  const r = await fetch(path, init);
  if (r.status === 401) { location = '/login'; throw new Error('login required'); }
  const data = await r.json().catch(() => ({}));
  if (!r.ok) { const m = data.detail || r.statusText; toast(m); throw new Error(m); }
  return data;
}

const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));

function md(src) {
  const blocks = [];
  let s = esc(src).replace(/```[\w-]*\n?([\s\S]*?)```/g, (_, code) => `\u0000${blocks.push(code) - 1}\u0000`);
  s = s.replace(/`([^`\n]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*\n]+)\*\*/g, '<b>$1</b>')
    .replace(/^#{1,6} (.*)$/gm, '<h4>$1</h4>')
    .replace(/^\s*(?:[-*]|\d+\.) (.*)$/gm, '<li>$1</li>')
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>')
    .replace(/\n/g, '<br>')
    .replace(/(<\/li>|<\/h4>)<br>/g, '$1');
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<pre><code>${blocks[i]}</code></pre>`);
}

const fmt = iso => iso ? new Date(iso).toLocaleString('en-IN', {timeZone: TZ, dateStyle: 'medium', timeStyle: 'short'}) : '';
const daysAgo = iso => Math.floor((Date.now() - new Date(iso)) / 864e5);

async function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) { await navigator.clipboard.writeText(text); return true; }
  const ta = document.createElement('textarea');  // LAN is plain http: no Clipboard API
  ta.value = text; document.body.append(ta); ta.select();
  const ok = document.execCommand('copy'); ta.remove(); return ok;
}

// ---------- navigation ----------
const loaders = {chat: loadChat, projects: loadProjects, ideas: loadIdeas, memory: loadMemory, today: loadToday};
function show(v) {
  if (!loaders[v]) v = 'chat';
  document.querySelectorAll('nav button').forEach(b => b.classList.toggle('on', b.dataset.v === v));
  document.querySelectorAll('main > section').forEach(s => s.classList.toggle('on', s.id === v));
  history.replaceState(null, '', '#' + v);
  loaders[v]().catch(e => console.error(e));
}

// ---------- chat ----------
async function refreshConvs() {
  const list = await api('/api/conversations');
  $('#convs').innerHTML = list.map(c =>
    `<option value="${c.id}" ${c.id === conv ? 'selected' : ''}>${esc(c.title)} · ${fmt(c.updated_at)}</option>`).join('');
  return list;
}
async function loadChat() {
  if (conv) return;
  const list = await refreshConvs();
  if (list[0] && Date.now() - new Date(list[0].updated_at) < 6 * 3600e3) await openConv(list[0].id);
  else await newConv();
}
async function newConv() {
  $('#msgs').innerHTML = '<div class="tool">Waking up…</div>';
  const c = await api('/api/conversations', {method: 'POST'});
  conv = c.id; $('#msgs').innerHTML = '';
  addMsg('model', c.greeting); refreshConvs(); refreshStatus();
}
async function openConv(id) {
  conv = id; $('#msgs').innerHTML = '';
  for (const m of await api(`/api/conversations/${id}/messages`)) {
    for (const t of (m.meta.tools || [])) addTool({name: t.name, args: t.args}, t.result);
    addMsg(m.role === 'user' ? 'user' : 'model', m.content, m.meta);
  }
  refreshConvs();
}
function el(cls, html) { const d = document.createElement('div'); d.className = cls; d.innerHTML = html; return d; }
function place(node, before) {
  before ? $('#msgs').insertBefore(node, before) : $('#msgs').append(node);
  node.scrollIntoView({block: 'end'}); return node;
}
function addMsg(role, text, meta = {}, before) {
  const d = el('msg ' + role, role === 'user' ? esc(text).replace(/\n/g, '<br>') : md(text));
  const mems = meta.memories || [];
  const chips = [];
  if (mems.length) chips.push(`<button class="chip" onclick="this.parentNode.nextElementSibling.hidden^=1">🧠 ${mems.length} memories</button>`);
  if (meta.model) chips.push(`<span class="chip">${esc(meta.model)}</span>`);
  if (chips.length) d.append(el('meta', chips.join('')));
  if (mems.length) { const l = el('small muted', mems.map(m => '• ' + esc(m.text)).join('<br>')); l.hidden = true; d.append(l); }
  return place(d, before);
}
function toolLabel(ev) {
  const a = ev.args || {};
  if (ev.name === 'ask_claude_code') return `🔧 Claude Code in <b>${esc(a.project)}</b>: ${esc((a.prompt || '').slice(0, 120))}`;
  if (ev.name === 'web_search') return `🔎 ${esc(a.query)}`;
  return `🔧 ${esc(ev.name)}`;
}
function setToolResult(d, result) {
  const body = result.error ? '⚠️ ' + result.error : (result.result ?? result.answer ?? JSON.stringify(result, null, 1));
  d.insertAdjacentHTML('beforeend', `<details><summary>result</summary><pre>${esc(String(body).slice(0, 8000))}</pre></details>`);
}
function addTool(ev, result, before) {
  const d = el('tool', toolLabel(ev));
  if (result !== undefined) setToolResult(d, result);
  return place(d, before);
}
async function send(e) {
  e.preventDefault();
  const text = $('#input').value.trim();
  if (!text || busy || !conv) return;
  busy = true; $('#sendbtn').disabled = true; $('#input').value = '';
  addMsg('user', text);
  const status = place(el('tool', 'thinking…'));
  const pending = [];
  try {
    const r = await fetch('/api/chat', {method: 'POST', headers: {'Content-Type': 'application/json'},
                                        body: JSON.stringify({conversation_id: conv, text})});
    if (r.status === 401) { location = '/login'; return; }
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
    const reader = r.body.getReader(), dec = new TextDecoder();
    let buf = '';
    for (;;) {
      const {value, done} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      let i;
      while ((i = buf.indexOf('\n\n')) >= 0) {
        const chunk = buf.slice(0, i); buf = buf.slice(i + 2);
        if (!chunk.startsWith('data: ')) continue;
        const ev = JSON.parse(chunk.slice(6));
        if (ev.type === 'tool') pending.push(addTool(ev, undefined, status));
        else if (ev.type === 'tool_result') { const d = pending.shift(); if (d) setToolResult(d, ev.result); }
        else if (ev.type === 'answer') addMsg('model', ev.text, ev, status);
        else if (ev.type === 'error') addMsg('model', '⚠️ ' + ev.text, {}, status);
      }
    }
  } catch (err) {
    addMsg('model', `⚠️ ${err.message}. If Claude Code was running, reopen this chat in a minute: the answer is saved when it finishes.`, {}, status);
  } finally {
    status.remove(); busy = false; $('#sendbtn').disabled = false; refreshStatus(); refreshConvs();
  }
}

// ---------- projects ----------
const FIELDS = ['goal', 'next_step', 'notes', 'path', 'main_files'];
async function loadProjects() {
  const ps = await api('/api/projects');
  $('#plist').innerHTML = ps.map(p => `<div class="card click" onclick="openProject(${p.id})">
      <div class="row"><b>${esc(p.name)}</b><span class="st ${esc(p.status)}">${esc(p.status)}</span></div>
      <div class="muted">${esc(p.next_step || p.goal || 'No next step yet')}</div>
      <div class="muted small">${p.last_activity_at ? daysAgo(p.last_activity_at) + 'd since activity' : 'no activity'}${p.path ? ' · 📁 ' + esc(p.path) : ''}</div>
    </div>`).join('') || '<p class="muted">No projects yet. They appear after the first Claude history sync, or add one.</p>';
}
async function openProject(id) {
  const p = await api('/api/projects/' + id);
  const field = f => `<label>${f.replace('_', ' ')}</label>` + (f === 'notes'
    ? `<textarea id="pf_${f}" rows="5">${esc(p[f])}</textarea>` : `<input id="pf_${f}" value="${esc(p[f])}">`);
  const list = (items, render) => items.map(render).join('') || '<p class="muted">None yet.</p>';
  $('#pbody').innerHTML = `
    <div class="row"><h2 style="margin:0">${esc(p.name)}</h2><button class="ghost" onclick="closeProject()">✕ Close</button></div>
    <label>status</label><select id="pf_status">${['active', 'stalled', 'paused', 'done'].map(s =>
      `<option ${s === p.status ? 'selected' : ''}>${s}</option>`).join('')}</select>
    ${FIELDS.map(field).join('')}
    <div class="row" style="margin-top:12px"><span></span>
      <button class="ghost" onclick="resume(${p.id})">📋 Resume context</button>
      <button class="btn" onclick="saveProject(${p.id})">Save</button></div>
    <div id="resumebox"></div>
    <h2>Decisions</h2>${list(p.decisions, d => `<div class="card"><div>${esc(d.text)}</div>
      <div class="muted small">${esc(d.why)} · ${fmt(d.decided_at)} · ${esc(d.source)}</div></div>`)}
    <h2>Links</h2>${list(p.links, l => `<div class="card"><a href="${esc(l.url)}" target="_blank" rel="noopener">${esc(l.title || l.url)}</a>
      <div class="small">${md(l.summary)}</div></div>`)}
    <h2>Memories</h2>${list(p.memories, m => `<div class="card small">${esc(m.text)}</div>`)}
    <h2>Claude Code runs</h2>${list(p.runs, r => `<div class="card"><div>${esc(r.prompt)}</div>
      <div class="muted small">${fmt(r.created_at)} · ${r.duration_s}s · exit ${r.exit_code} · changed: ${esc(JSON.parse(r.files_changed || '[]').join(', ') || 'none')}</div>
      <details><summary class="small">output</summary><pre class="small" style="white-space:pre-wrap">${esc(r.output)}</pre></details></div>`)}`;
  $('#pdetail').classList.add('on');
}
function closeProject() { $('#pdetail').classList.remove('on'); loadProjects(); }
async function saveProject(id) {
  const body = {status: $('#pf_status').value};
  for (const f of FIELDS) body[f] = $('#pf_' + f).value;
  await api('/api/projects/' + id, {method: 'PATCH', body});
  toast('Saved', true);
}
async function resume(id) {
  const {text} = await api(`/api/projects/${id}/resume`);
  const ok = await copyText(text).catch(() => false);
  $('#resumebox').innerHTML = `<label>${ok ? 'Copied to clipboard ✓' : 'Copy this:'}</label><textarea rows="10" readonly>${esc(text)}</textarea>`;
}
async function newProject() {
  const name = prompt('Project name');
  if (!name) return;
  const path = prompt('Folder path on this machine (optional)') || undefined;
  const r = await api('/api/projects', {method: 'POST', body: {name, path}});
  await loadProjects(); openProject(r.id);
}

// ---------- ideas ----------
async function loadIdeas() {
  const st = $('#istatus').value;
  const ideas = await api('/api/ideas?status=' + st);
  $('#ilist').innerHTML = ideas.map(i => `<div class="card"><div>${esc(i.text)}</div>
      <div class="muted small">${esc(i.tags)} ${fmt(i.created_at)}</div>
      ${st === 'inbox' ? `<div class="row" style="margin-top:8px"><span></span>
        <button class="ghost" onclick="related(${i.id})">Related</button>
        <button class="ghost" onclick="setIdea(${i.id}, 'dropped')">Drop</button>
        <button class="btn" onclick="promote(${i.id}, this)">Promote →</button></div>` : ''}
      ${st === 'dropped' ? `<div class="row" style="margin-top:8px"><span></span>
        <button class="ghost" onclick="setIdea(${i.id}, 'inbox')">Restore</button></div>` : ''}
      <div id="rel${i.id}" class="small" style="margin-top:6px"></div></div>`).join('') || '<p class="muted">Nothing here.</p>';
}
async function addIdea(e) {
  e.preventDefault();
  const text = $('#ideatext').value.trim();
  if (!text) return;
  const r = await api('/api/ideas', {method: 'POST', body: {text}});
  $('#ideatext').value = '';
  await loadIdeas();
  if (r.related?.length) showRelated(r.id, r.related);
}
async function setIdea(id, status) { await api('/api/ideas/' + id, {method: 'PATCH', body: {status}}); loadIdeas(); }
async function promote(id, btn) {
  btn.disabled = true; btn.textContent = 'Planning…';
  try {
    const r = await api(`/api/ideas/${id}/promote`, {method: 'POST'});
    show('projects'); openProject(r.project_id);
  } finally { btn.disabled = false; btn.textContent = 'Promote →'; }
}
async function related(id) { showRelated(id, await api(`/api/ideas/${id}/related`)); }
function showRelated(id, items) {
  const box = $('#rel' + id);
  if (!box) return;
  box.innerHTML = items.length
    ? '<b>Related:</b> ' + items.map(r => `${r.type === 'project' ? '📁' : '💡'} ${esc(r.text.slice(0, 80))} (${Math.round(r.score * 100)}%)`).join(' · ')
    : '<span class="muted">Nothing similar yet.</span>';
}

// ---------- memory ----------
async function loadMemory() {
  const [prof, mems] = await Promise.all([api('/api/profile'), api('/api/memories?q=' + encodeURIComponent($('#memq').value))]);
  if (document.activeElement !== $('#profile')) $('#profile').value = prof.content || '';
  $('#mlist').innerHTML = mems.map(m => `<div class="card"><div>${esc(m.text)}</div>
      <div class="row small muted"><span>${esc(m.kind)}${m.source ? ' · ' + esc(m.source) : ''}${m.score ? ' · ' + Math.round(m.score * 100) + '%' : ''}</span>
      <button class="ghost" onclick="editMemory(${m.id}, this)">Edit</button>
      <button class="ghost" onclick="delMemory(${m.id})">Delete</button></div></div>`).join('') || '<p class="muted">No memories yet.</p>';
}
async function editMemory(id, btn) {
  const cur = btn.closest('.card').firstElementChild.textContent;
  const text = prompt('Edit memory', cur);
  if (text && text !== cur) { await api('/api/memories/' + id, {method: 'PATCH', body: {text}}); loadMemory(); }
}
async function delMemory(id) {
  if (confirm('Delete this memory?')) { await api('/api/memories/' + id, {method: 'DELETE'}); loadMemory(); }
}
async function addMemory(e) {
  e.preventDefault();
  const text = $('#memnew').value.trim();
  if (!text) return;
  await api('/api/memories', {method: 'POST', body: {text}});
  $('#memnew').value = ''; loadMemory();
}
async function saveProfile() { await api('/api/profile', {method: 'PUT', body: {content: $('#profile').value}}); toast('Profile saved', true); }

// ---------- today ----------
function setBadge(rs) { const n = rs.filter(r => new Date(r.due_at) <= Date.now()).length; $('#duebadge').textContent = n || ''; }
async function loadToday() {
  const t = await api('/api/today');
  $('#rlist').innerHTML = t.reminders.map(r => `<div class="card row ${new Date(r.due_at) <= Date.now() ? 'due' : ''}">
      <span>${esc(r.text)}<br><span class="small muted">${fmt(r.due_at)}${r.project ? ' · ' + esc(r.project) : ''}</span></span>
      <button class="ghost" onclick="doneReminder(${r.id})">✓ Done</button></div>`).join('') || '<p class="muted">No open reminders.</p>';
  $('#work').innerHTML = t.work_on.map((p, i) => `<div class="card click" onclick="show('projects');openProject(${p.id})">
      <b>${i + 1}. ${esc(p.name)}</b> <span class="muted small">${esc(p.reasons.join(' · '))}</span>
      <div class="muted">${esc(p.next_step || '')}</div></div>`).join('') || '<p class="muted">No active projects.</p>';
  $('#recaps').innerHTML = t.recaps.map(r => `<div class="card"><div class="muted small">${esc(r.period)} · ${esc(r.period_start)}</div>${md(r.content)}</div>`).join('')
    || '<p class="muted">The first recap is written tonight at 02:00.</p>';
  $('#syncinfo').textContent = t.next_sync ? 'Next auto-sync ' + fmt(t.next_sync) : 'Not synced yet';
  setBadge(t.reminders);
}
async function doneReminder(id) { await api('/api/reminders/' + id, {method: 'PATCH', body: {done: true}}); loadToday(); }
async function addReminder(e) {
  e.preventDefault();
  const text = $('#remtext').value.trim(), when = $('#remwhen').value;
  if (!text || !when) return toast('Need text and a time');
  await api('/api/reminders', {method: 'POST', body: {text, when}});
  $('#remtext').value = ''; $('#remwhen').value = ''; loadToday();
}
async function syncNow(btn) {
  btn.disabled = true; btn.textContent = 'Syncing…';
  try {
    const s = await api('/api/sync', {method: 'POST'});
    toast(s.busy ? 'Sync already running' : `Synced ${s.processed}, unchanged ${s.skipped}, failed ${s.failed}, deferred ${s.deferred}`, true);
  } finally { btn.disabled = false; btn.textContent = 'Sync now'; loadToday(); }
}

// ---------- status ----------
async function refreshStatus() {
  const s = await api('/api/status');
  TZ = s.timezone;
  const out = Object.entries(s.exhausted);
  $('#model').textContent = (s.model || 'no model yet') + (out.length ? ` · ${out.length} out of quota` : '');
  $('#model').title = out.map(([m, t]) => `${m} until ${fmt(new Date(t * 1000).toISOString())}`).join('\n') || 'All models available';
}
async function logout() { await fetch('/logout', {method: 'POST'}); location = '/login'; }
async function pollDue() { try { setBadge((await api('/api/today')).reminders); } catch {} }

refreshStatus().finally(() => show(location.hash.slice(1) || 'chat'));
pollDue(); setInterval(pollDue, 60000);
</script>
</body>
</html>
```

- [ ] **Step 2: Run the test suite (nothing should regress)**

Run: `uv run pytest -q test_core.py`
Expected: all pass

- [ ] **Step 3: Smoke-render with a fake backend is not needed; real check happens in Task 10.** Commit:

```bash
git add static/index.html
git commit -m "feat: mobile-first single page UI"
```

---

### Task 10: README, first real run, systemd

**Files:**
- Create: `README.md`
- Modify: `.env` (append two keys; do NOT print or commit it)

- [ ] **Step 1: Add password and session secret to `.env`** (user chooses the password)

```bash
cd /mnt/newdisk/neel/ai-new-projects
grep -qi '^SESSION_SECRET=' .env || echo "SESSION_SECRET=$(python3 -c 'import secrets;print(secrets.token_hex(32))')" >> .env
grep -qi '^APP_PASSWORD=' .env || echo "APP_PASSWORD=CHANGE_ME" >> .env
```
Tell the user to replace `CHANGE_ME` with their own password.

- [ ] **Step 2: Write `README.md`**

````markdown
# Gemini Personal Assistant

A web app on your home network. Gemini is your assistant: it remembers projects, ideas and decisions, syncs from your Claude Code history every 3 h, and can run Claude Code inside a project folder to answer questions about the code.

## Setup
1. `.env` in this folder:
   ```
   gemini_api_key=...
   APP_PASSWORD=your-password
   SESSION_SECRET=<random hex>
   ```
   Optional overrides: `MODELS`, `EMBED_MODEL`, `SEARCH_MODELS`, `RPM_LIMIT`, `PORT` (8765), `TIMEZONE` (Asia/Kolkata), `CLAUDE_TIMEOUT` (600), `SYNC_MAX_CALLS` (30).
2. `uv sync`
3. `uv run main.py`
4. Open `http://localhost:8765` here, or `http://<this-machine-LAN-IP>:8765` on your phone (`hostname -I` shows the IP). If the phone can't connect: `sudo ufw allow 8765/tcp`.

## Using it
- Chat: "idea: …", "remind me Friday 5pm to …", "what should I work on?", "in key_manager, which providers do I use?" (runs Claude Code in that folder).
- Tell it where a project lives: "ai call bot is at /mnt/newdisk/neel/ai_call_bot, main file server.py".
- Claude Code runs with **full access** (`bypassPermissions`) in registered folders. Every run is logged on the project page with changed files.
- Quota: models are tried in order; per-minute limits skip a model for 60 s, per-day limits until midnight Pacific. The header chip shows the model in use.

## Start on boot (systemd user service)
```
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/assistant.service <<'EOF'
[Unit]
Description=Gemini personal assistant
After=network-online.target

[Service]
WorkingDirectory=/mnt/newdisk/neel/ai-new-projects
ExecStart=/home/neel/.local/bin/uv run main.py
Environment=PATH=/home/neel/.local/bin:/usr/local/bin:/usr/bin:/bin
Restart=on-failure

[Install]
WantedBy=default.target
EOF
systemctl --user daemon-reload
systemctl --user enable --now assistant
loginctl enable-linger $USER   # keep running when logged out
journalctl --user -u assistant -f
```

## Files
`main.py` app/routes · `agent.py` chat loop · `tools.py` Gemini tools · `memory.py` memory · `llm.py` Gemini + fallback · `claude_runner.py` Claude Code · `jobs.py` sync/nightly · `db.py` schema · `static/` UI · `test_core.py` tests (`uv run pytest -q`).
Data: `assistant.db` (+ nightly copies in `backups/`, last 7).
````

- [ ] **Step 3: Full test run**

Run: `uv run pytest -q`
Expected: all pass

- [ ] **Step 4: Live smoke test** (needs real network — run outside any sandbox; uses a few Gemini requests)

```bash
uv run main.py &   # wait until "Uvicorn running on http://0.0.0.0:8765"
PW=$(grep -i '^APP_PASSWORD=' .env | cut -d= -f2-)
curl -s -c /tmp/ck -H 'Content-Type: application/json' -d "{\"password\":\"$PW\"}" localhost:8765/login
curl -s -b /tmp/ck -X POST localhost:8765/api/conversations            # expect {"id":1,"greeting":"..."}
curl -s -N -b /tmp/ck -H 'Content-Type: application/json' \
  -d '{"conversation_id":1,"text":"Save an idea: a dashboard for GPU prices"}' localhost:8765/api/chat
# expect data: {"type": "tool", "name": "save_idea"...} ... data: {"type": "answer"...}
curl -s -b /tmp/ck localhost:8765/api/status                            # model = gemini-3.5-flash-lite
tail -n 20 assistant.log                                                # sync progress, no tracebacks
```
Then open `http://<LAN-IP>:8765` on a phone (or browser devtools at 390 px width): log in, check all five tabs render without horizontal scroll, send a message, open a synced project, press "Resume context".

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: setup, usage and systemd service"
```
