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
CREATE TABLE IF NOT EXISTS claude_sessions(
  id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  claude_session_id TEXT, status TEXT NOT NULL DEFAULT 'queued', pending_message TEXT,
  resume_at TEXT, resume_estimated INTEGER DEFAULT 0, resume_scheduled INTEGER DEFAULT 0,
  last_error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, last_activity_at TEXT);
CREATE TABLE IF NOT EXISTS claude_events(
  id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL REFERENCES claude_sessions(id) ON DELETE CASCADE,
  seq INTEGER NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS claude_events_session ON claude_events(session_id, seq);
CREATE TABLE IF NOT EXISTS services(
  id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, script_path TEXT, tmux_session TEXT NOT NULL UNIQUE,
  cwd TEXT, venv TEXT, command TEXT, port INTEGER, depends_on TEXT DEFAULT '[]', gpu_mem_gb REAL,
  notes TEXT, managed INTEGER DEFAULT 1, should_run INTEGER DEFAULT 0,
  project_id INTEGER REFERENCES projects(id) ON DELETE SET NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS service_events(
  id INTEGER PRIMARY KEY, service_id INTEGER REFERENCES services(id) ON DELETE CASCADE, kind TEXT NOT NULL,
  detail TEXT DEFAULT '', created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS service_changes(
  id INTEGER PRIMARY KEY, service_id INTEGER REFERENCES services(id) ON DELETE CASCADE, request TEXT NOT NULL,
  status TEXT NOT NULL, claude_session INTEGER, backup_dir TEXT, manifest TEXT, result TEXT, diff TEXT DEFAULT '',
  flagged INTEGER DEFAULT 0, explanation TEXT DEFAULT '', created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS pending_actions(
  id INTEGER PRIMARY KEY, kind TEXT NOT NULL, service TEXT, payload TEXT DEFAULT '{}', created_by TEXT DEFAULT 'gemini',
  status TEXT NOT NULL DEFAULT 'pending', result TEXT, created_at TEXT NOT NULL, decided_at TEXT);
CREATE TABLE IF NOT EXISTS repos(
  id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, url TEXT NOT NULL, path TEXT NOT NULL UNIQUE,
  venv_python TEXT NOT NULL, default_branch TEXT NOT NULL, entry TEXT DEFAULT 'main',
  suite_status TEXT DEFAULT 'none', suite_branch TEXT, suite_error TEXT, known_failures TEXT DEFAULT '[]',
  env_status TEXT, env_error TEXT, last_pull TEXT,
  project_id INTEGER REFERENCES projects(id) ON DELETE SET NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS repo_tasks(
  id INTEGER PRIMARY KEY, repo_id INTEGER NOT NULL REFERENCES repos(id) ON DELETE CASCADE, request TEXT NOT NULL,
  branch TEXT, status TEXT NOT NULL, base_sha TEXT, head_sha TEXT, before_run INTEGER, after_run INTEGER,
  regressions TEXT DEFAULT '[]', new_tests TEXT DEFAULT '[]', fixed_tests TEXT DEFAULT '[]', new_failing TEXT DEFAULT '[]',
  summary TEXT, claude_result TEXT, diff_stat TEXT, claude_session INTEGER, error TEXT, env_warning TEXT,
  merge_prev_sha TEXT, merge_sha TEXT, merge_check TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS test_runs(
  id INTEGER PRIMARY KEY, repo_id INTEGER NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
  task_id INTEGER REFERENCES repo_tasks(id) ON DELETE SET NULL, ref TEXT, sha TEXT, kind TEXT NOT NULL,
  passed INTEGER DEFAULT 0, failed INTEGER DEFAULT 0, errors INTEGER DEFAULT 0, skipped INTEGER DEFAULT 0,
  results TEXT DEFAULT '{}', duration_s REAL, output_tail TEXT, created_at TEXT NOT NULL);
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
