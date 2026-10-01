"""Background work: Claude Code history sync, nightly reflection/recaps/backup, and the scheduler loop."""
import asyncio
import email.utils
import hashlib
import json
import logging
import sqlite3
import tempfile
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path

import httpx
import numpy as np
from google.genai import errors, types

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

def _ignored_cwd(cwd: str) -> bool:
    p = Path(cwd).resolve()  # scratch sessions in temp dirs or $HOME are not projects
    return (p == Path.home().resolve() or p.is_relative_to(Path(tempfile.gettempdir()).resolve())
            or p.is_relative_to(config.ROOT / "workspace"))  # the assistant's own repo clones


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
    if not turns or not cwd or _ignored_cwd(cwd):
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
        quota_out, api_err = False, None
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
            except (AllModelsExhausted, errors.APIError, httpx.HTTPError) as e:
                quota_out, api_err = True, f"{type(e).__name__}: {str(e)[:200]}"
                stats["deferred"] += 1
                db.kv_set(con, "sync_error", api_err)
                continue  # not marked: retried next run
            except Exception:
                log.exception("sync failed for %s", f)
                stats["failed"] += 1  # marked below so a bad file isn't retried until it changes
            con.execute("INSERT INTO sync_state(file_path, sha256, synced_at) VALUES(?, ?, ?) "
                        "ON CONFLICT(file_path) DO UPDATE SET sha256=excluded.sha256, synced_at=excluded.synced_at",
                        (str(f), digest, db.now()))
        if stats["processed"] and not api_err:
            db.kv_set(con, "sync_error", "")
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


def prune_claude_events(con, now: datetime) -> None:
    """Shrink big transcript payloads older than 14 days to a short preview."""
    con.execute("UPDATE claude_events SET payload=json_object('truncated', 1, 'type', kind, "
                "'preview', substr(payload, 1, 1000)) WHERE length(payload) > 1000 AND created_at < ?",
                (db.to_utc(now - timedelta(days=14)),))


async def nightly(con, llm, backup_dir: Path, today: date) -> None:
    yesterday = today - timedelta(days=1)
    steps = [
        ("profile", rewrite_profile(con, llm)),
        ("daily recap", write_recap(con, llm, "daily", yesterday, yesterday)),
    ]
    if yesterday.weekday() == 6:  # Sunday just ended
        steps.append(("weekly recap", write_recap(con, llm, "weekly", yesterday - timedelta(days=6), yesterday)))
    try:
        await backfill_embeddings(con, llm)  # before merge so fresh rows are deduped too
    except Exception:
        log.exception("nightly backfill failed")
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
        prune_claude_events(con, datetime.now(timezone.utc))
    except Exception:
        log.exception("nightly prune failed")
    try:
        backup(con, backup_dir, today)
    except Exception:
        log.exception("nightly backup failed")


# ---------- scheduler ----------

async def check_clock(con) -> float | None:
    """Compare the local clock with Google's Date header; store the offset (seconds)."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.head("https://generativelanguage.googleapis.com/")
        server = email.utils.parsedate_to_datetime(r.headers["date"])
    except Exception:
        log.warning("clock check failed", exc_info=True)
        return None
    offset = (server - datetime.now(timezone.utc)).total_seconds()
    db.kv_set(con, "clock_offset_s", f"{offset:.1f}")
    if abs(offset) > 120:
        log.warning("system clock differs from Google time by %.0f s", offset)
    return offset


async def tick(con, llm, now: datetime | None = None, sessions=None, services=None) -> None:
    now = now or datetime.now(timezone.utc)
    if sessions is not None:  # first: resumes must not wait behind a long sync
        if db.to_utc(now) >= (db.kv_get(con, "next_clock_check") or ""):
            db.kv_set(con, "next_clock_check", db.to_utc(now + timedelta(hours=1)))
            await check_clock(con)
        await sessions.fire_due_resumes()
    if services is not None:
        try:
            await services.health_tick()
        except Exception:
            log.exception("service health tick failed")
    if db.to_utc(now) >= (db.kv_get(con, "next_sync") or ""):
        stats = await sync_claude_history(con, llm, config.CLAUDE_HISTORY_DIR, config.SYNC_MAX_CALLS)
        delay = timedelta(minutes=10) if stats.get("deferred") or stats.get("busy") else timedelta(hours=3)
        db.kv_set(con, "next_sync", db.to_utc(now + delay))
    local = now.astimezone(config.TZ)
    if local.hour >= 2 and db.kv_get(con, "last_nightly") != local.date().isoformat():
        db.kv_set(con, "last_nightly", local.date().isoformat())  # set first: a crash must not loop every minute
        await nightly(con, llm, config.ROOT / "backups", local.date())


async def scheduler(con, llm, sessions=None, services=None) -> None:
    while True:
        try:
            await tick(con, llm, sessions=sessions, services=services)
        except Exception:
            log.exception("scheduler tick failed")
        await asyncio.sleep(60)
