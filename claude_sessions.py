"""Live Claude Code sessions: one streaming `claude` process per project, watchable and steerable.

CLI contract verified 2026-09-30 (Claude Code 2.1.284): with stream-json in/out the process stays
alive across turns; each stdin line is a user turn; each turn ends with a `result` event; every turn
also emits `rate_limit_event.rate_limit_info` = {status, resetsAt (epoch s), unifiedWindows{...}}.
"""
import asyncio
import contextlib
import json
import logging
import os
import re
import signal
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import config
import db

log = logging.getLogger("claude_sessions")
ALLOWED = ("allowed", "allowed_warning")
LIMIT_TEXT = re.compile(r"usage limit|limit reached", re.I)
ESTIMATE = timedelta(hours=5)
MAX_PAYLOAD = 64_000
EVENT_SOFT_CAP = 5000  # beyond this, assistant/user payloads are stored truncated
_CLOSE = object()


def parse_reset(text: str, now: datetime) -> datetime | None:
    """Reset time from a limit message: '...|<epoch>' or 'resets 7pm (UTC)'. None if absent."""
    m = re.search(r"\|(\d{10})\b", text or "")
    if m:
        return datetime.fromtimestamp(int(m.group(1)), timezone.utc)
    m = re.search(r"resets?\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)(?:\s*\(([^)]+)\))?", text or "", re.I)
    if not m:
        return None
    try:
        tz = ZoneInfo(m.group(4).strip()) if m.group(4) else timezone.utc
    except (ZoneInfoNotFoundError, ValueError):
        tz = timezone.utc
    hour = int(m.group(1)) % 12 + (12 if m.group(3).lower() == "pm" else 0)
    local = now.astimezone(tz)
    minute = int(m.group(2) or 0)
    if not 1 <= int(m.group(1)) <= 12 or minute > 59:
        return None
    cand = local.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if cand <= local:
        cand += timedelta(days=1)
    return cand.astimezone(timezone.utc)


def _proc_table() -> dict[int, tuple[int, int]]:
    """pid -> (ppid, session id) from /proc/<pid>/stat."""
    t = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                f = Path(f"/proc/{d}/stat").read_text().rsplit(")", 1)[1].split()
            except (OSError, IndexError):
                continue
            t[int(d)] = (int(f[1]), int(f[3]))  # after ')': state ppid pgrp session
    return t


def _tree_sids(root: int) -> set[int]:
    """Session ids of root and all its descendants (recorded while root is alive)."""
    t = _proc_table()
    kids: dict[int, list[int]] = {}
    for pid, (ppid, _) in t.items():
        kids.setdefault(ppid, []).append(pid)
    todo, sids = [root], set()
    while todo:
        p = todo.pop()
        if p in t:
            sids.add(t[p][1])
            todo += kids.get(p, [])
    sids.discard(os.getsid(0))  # never our own session
    return sids


def _kill_sids(sids: set[int]) -> None:
    me = os.getpid()
    for _ in range(2):  # second pass catches a fork racing the first
        for pid, (_, sid) in _proc_table().items():
            if sid in sids and pid != me:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.kill(pid, signal.SIGKILL)


async def _git_status(path: str) -> set[str] | None:
    proc = await asyncio.create_subprocess_exec(
        "git", "-C", path, "status", "--porcelain",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    out, _ = await proc.communicate()
    return set(out.decode().splitlines()) if proc.returncode == 0 else None


def _local(dt: datetime) -> str:
    return dt.astimezone(config.TZ).strftime("%a %d %b %H:%M")


class Session:
    """One Claude conversation (a claude_sessions row) and, while live, its process."""

    def __init__(self, mgr: "SessionManager", row: dict, project: dict):
        self.mgr, self.con, self.id, self.project = mgr, mgr.con, row["id"], project
        self.claude_session_id = row["claude_session_id"]
        self.turns: asyncio.Queue = asyncio.Queue()
        self.results: asyncio.Queue = asyncio.Queue()
        self.watchers: set[asyncio.Queue] = set()
        self.proc = None
        self.sids: set[int] = set()
        self.readers: list[asyncio.Task] = []
        self.stderr_tail = b""
        self.turn_rate: dict = {}
        self.inflight: asyncio.Future | None = None
        self.accepting = True
        self.seq = db.one(self.con, "SELECT COALESCE(MAX(seq), 0) AS s FROM claude_events WHERE session_id=?",
                          self.id)["s"]
        self._status("queued")
        self.task = asyncio.create_task(self._run())

    # ---------- events ----------
    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self.watchers.add(q)
        return q

    def _emit(self, kind: str, payload) -> None:
        self.seq += 1
        raw = payload if isinstance(payload, str) else json.dumps(payload, default=str)
        cap = 1000 if self.seq > EVENT_SOFT_CAP and kind in ("assistant", "user") else MAX_PAYLOAD
        if len(raw) > cap:
            raw = json.dumps({"truncated": True, "type": kind, "preview": raw[:cap]})
        db.insert(self.con, "claude_events", session_id=self.id, seq=self.seq, kind=kind, payload=raw,
                  created_at=db.now())
        ev = {"seq": self.seq, "kind": kind, "payload": raw}
        for q in list(self.watchers):
            q.put_nowait(ev)

    def _status(self, status: str, **fields) -> None:
        db.update(self.con, "claude_sessions", self.id, status=status, updated_at=db.now(), **fields)
        self._emit("status", {"status": status, **{k: fields[k] for k in ("resume_at", "last_error") if k in fields}})

    # ---------- turns ----------
    def submit(self, text: str) -> asyncio.Future:
        fut = asyncio.get_running_loop().create_future()
        self._emit("user_turn", {"text": text})
        self.turns.put_nowait((text, fut))
        return fut

    def close_when_idle(self) -> None:
        self.turns.put_nowait((_CLOSE, None))

    def _fail_pending(self, res: dict) -> None:
        futs = [self.inflight] if self.inflight else []
        while not self.turns.empty():
            futs.append(self.turns.get_nowait()[1])
        for f in futs:
            if f is not None and not f.done():
                f.set_result(res)

    async def _run(self) -> None:
        try:
            while True:
                try:
                    text, fut = await asyncio.wait_for(self.turns.get(), self.mgr.idle_s)
                except asyncio.TimeoutError:
                    if not self.turns.empty():  # a turn slipped in as the timer fired
                        continue
                    text, fut = _CLOSE, None
                if text is _CLOSE:
                    self.accepting = False
                    self._status("closed")
                    return
                self.inflight = fut
                if self.mgr.slots.locked():
                    self._status("queued")
                async with self.mgr.slots:  # slot held per turn: idle sessions don't block others
                    try:
                        res = await self._turn(text)
                    except Exception as e:  # not CancelledError: stop() relies on it
                        log.exception("session %s: turn failed", self.id)
                        res = self._end("failed", f"Internal error: {type(e).__name__}: {e}")
                self.inflight = None
                if not fut.done():
                    fut.set_result(res)
                if res.get("final"):
                    self.accepting = False
                    return
        finally:
            self.accepting = False
            self._fail_pending({"result": "Session closed before this turn ran; send it again.",
                                "session": self.id, "files_changed": [], "exit_code": None,
                                "timed_out": False, "final": True})
            await self._close_proc()
            for q in list(self.watchers):
                q.put_nowait(None)
            self.mgr._forget(self)

    async def _turn(self, text: str) -> dict:
        path = self.project.get("path")
        if not path or not Path(path).is_dir():
            return self._end("failed", f"project folder {path!r} does not exist")
        await self._ensure_proc()
        before = await _git_status(path)
        self.turn_rate = {}
        self._status("running")
        start = time.monotonic()
        try:
            line = json.dumps({"type": "user", "message": {"role": "user", "content": text}}) + "\n"
            self.proc.stdin.write(line.encode())
            await self.proc.stdin.drain()
            ev = await asyncio.wait_for(self.results.get(), self.mgr.timeout)
        except asyncio.TimeoutError:
            self._killpg()
            return self._end("failed", f"Claude Code timed out after {self.mgr.timeout}s and was stopped.",
                             timed_out=True)
        except (BrokenPipeError, ConnectionResetError):
            ev = None
        if ev is None:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self.proc.wait(), 5)
            tail = self.stderr_tail.decode(errors="replace")[-1500:]
            return self._end("failed", f"Claude Code exited without a result (exit {self.proc.returncode}). {tail}".strip())
        sid = ev.get("session_id")
        if sid:
            self.claude_session_id = sid
            db.update(self.con, "claude_sessions", self.id, claude_session_id=sid)
            db.update(self.con, "projects", self.project["id"], claude_session_id=sid, last_activity_at=db.now())
        out = str(ev.get("result", ""))
        if self._is_limit(ev):
            return self._pause(text, out)
        after = await _git_status(path)
        # ponytail: files already dirty before the turn and edited again won't show; diff content if that matters
        changed = sorted(after - before) if before is not None and after is not None else []
        code = 1 if ev.get("is_error") else 0
        db.insert(self.con, "claude_runs", project_id=self.project["id"], prompt=text, output=out, session_id=sid,
                  files_changed=json.dumps(changed), exit_code=code,
                  duration_s=round(time.monotonic() - start, 1), created_at=db.now())
        self._status("idle", last_activity_at=db.now())
        return {"result": out, "session": self.id, "files_changed": changed, "exit_code": code, "timed_out": False}

    def _is_limit(self, ev: dict) -> bool:
        if not ev.get("is_error"):
            return False
        rejected = self.turn_rate.get("status", "allowed") not in ALLOWED
        return rejected or bool(LIMIT_TEXT.search(str(ev.get("result", ""))))

    def _pause(self, text: str, message: str) -> dict:
        now = self.mgr.now()
        rate = self.turn_rate
        reset = None
        if rate.get("resetsAt") and rate.get("status") not in ALLOWED:
            reset = datetime.fromtimestamp(int(rate["resetsAt"]), timezone.utc)
        reset = reset or parse_reset(message, now)
        estimated = reset is None or reset <= now  # stale epoch: estimate instead of looping
        if estimated:
            reset = now + ESTIMATE
        self._status("paused_limit", pending_message=text, resume_at=db.to_utc(reset),
                     resume_estimated=int(estimated), resume_scheduled=0, last_error=message[:500])
        note = " (estimated)" if estimated else ""
        return {"result": f"Claude usage limit reached; resets {_local(reset)}{note} ({config.TZ.key}). "
                          "Ask the user whether to resume it then (schedule_claude_resume).",
                "session": self.id, "files_changed": [], "exit_code": None, "timed_out": False,
                "final": True, "limit": True}

    def _end(self, status: str, message: str, timed_out: bool = False) -> dict:
        self._status(status, last_error=message[:2000])
        return {"result": message, "session": self.id, "files_changed": [], "exit_code": None,
                "timed_out": timed_out, "final": True}

    # ---------- process ----------
    async def _ensure_proc(self) -> None:
        if self.proc and self.proc.returncode is None:
            return
        self._status("starting")
        cmd = [self.mgr.claude_bin, "-p", "--input-format", "stream-json", "--output-format", "stream-json",
               "--verbose", "--permission-mode", "bypassPermissions"]  # full access: the user's explicit choice
        if self.claude_session_id:
            cmd += ["--resume", self.claude_session_id]
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        self.results = asyncio.Queue()
        self.stderr_tail = b""
        self.proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=self.project["path"], env=env, start_new_session=True, limit=16 * 1024 * 1024,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        self.readers = [asyncio.create_task(self._read_stdout(self.proc, self.results)),
                        asyncio.create_task(self._read_stderr(self.proc))]

    async def _read_stdout(self, proc, results: asyncio.Queue) -> None:
        try:
            async for raw in proc.stdout:
                line = raw.decode(errors="replace").strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    self._emit("raw", line[:2000])
                    continue
                kind = ev.get("type") or "unknown"
                if kind == "system" and str(ev.get("subtype", "")).startswith("hook"):
                    continue  # hook chatter: not useful in the transcript
                if kind == "system" and ev.get("subtype") == "init" and ev.get("session_id"):
                    self.claude_session_id = ev["session_id"]  # known even if the turn later crashes
                    db.update(self.con, "claude_sessions", self.id, claude_session_id=ev["session_id"])
                if kind == "rate_limit_event":
                    self.turn_rate = ev.get("rate_limit_info") or {}
                    db.kv_set(self.con, "claude_usage", json.dumps(self.turn_rate))
                self._emit(kind, line)
                if kind in ("user", "assistant", "result"):
                    self._snapshot()
                if kind == "result":
                    results.put_nowait(ev)
        except Exception:
            log.exception("session %s: stdout reader failed", self.id)
        finally:
            results.put_nowait(None)

    async def _read_stderr(self, proc) -> None:
        with contextlib.suppress(Exception):
            async for raw in proc.stderr:
                self.stderr_tail = (self.stderr_tail + raw)[-4000:]

    def _snapshot(self) -> None:
        """Remember the sessions of claude and its descendants: Claude's Bash tool runs commands in their own
        session (SID=PGID=pid), which a process-group kill misses and which outlive claude once reparented."""
        if self.proc and self.proc.returncode is None:
            self.sids |= _tree_sids(self.proc.pid)

    def _killpg(self) -> None:
        if self.proc:
            self._snapshot()
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(self.proc.pid, signal.SIGKILL)
            _kill_sids(self.sids | {self.proc.pid})  # everything Claude's tools started, in any session

    async def _close_proc(self) -> None:
        if not self.proc:
            return
        if self.proc.returncode is None:
            self._snapshot()
            with contextlib.suppress(Exception):
                self.proc.stdin.close()
            try:
                await asyncio.wait_for(self.proc.wait(), 5)
            except asyncio.TimeoutError:
                self._killpg()
                await self.proc.wait()
        if self.sids:  # leftovers in Claude's tool sessions, e.g. a dev server it backgrounded
            _kill_sids(self.sids)
        for t in self.readers:
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await asyncio.wait_for(t, 2)

    async def stop(self) -> None:
        if not self.accepting:  # already ending (final outcome/closing): let cleanup finish, keep its status
            with contextlib.suppress(asyncio.CancelledError):
                await self.task
            return
        self.accepting = False
        self._status("stopped")
        # resolve waiters first, so the worker's cleanup doesn't answer them with a generic message
        self._fail_pending({"result": "Stopped by user.", "session": self.id, "files_changed": [],
                            "exit_code": None, "timed_out": False, "final": True})
        self._killpg()
        self.task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self.task


class SessionManager:
    def __init__(self, con, claude_bin: str = "claude", timeout: int = 600, idle_s: float = 1800,
                 max_running: int = 3, clock=time.time):
        self.con, self.claude_bin, self.timeout, self.idle_s, self.clock = con, claude_bin, timeout, idle_s, clock
        self.slots = asyncio.Semaphore(max_running)
        self.live: dict[int, Session] = {}
        self.locks: dict[int, asyncio.Lock] = {}
        con.execute("UPDATE claude_sessions SET status='stopped', updated_at=? "
                    "WHERE status IN ('queued', 'starting', 'running', 'idle')", (db.now(),))

    def now(self) -> datetime:
        off = float(db.kv_get(self.con, "clock_offset_s", "0") or 0)
        return datetime.fromtimestamp(self.clock() + (off if abs(off) > 120 else 0), timezone.utc)

    def _forget(self, s: Session) -> None:
        if self.live.get(s.project["id"]) is s:
            del self.live[s.project["id"]]

    def live_by_id(self, session_id: int) -> Session | None:
        return next((s for s in self.live.values() if s.id == session_id), None)

    def _new_row(self, project: dict, claude_session_id: str | None) -> dict:
        rid = db.insert(self.con, "claude_sessions", project_id=project["id"], claude_session_id=claude_session_id,
                        status="queued", created_at=db.now(), updated_at=db.now(), last_activity_at=db.now())
        return db.one(self.con, "SELECT * FROM claude_sessions WHERE id=?", rid)

    async def submit(self, project: dict, text: str, follow_up: bool = True, row: dict | None = None) -> asyncio.Future:
        lock = self.locks.setdefault(project["id"], asyncio.Lock())
        async with lock:
            s = self.live.get(project["id"])
            if s and not s.accepting:
                await asyncio.wait([s.task])
                s = None
            if s and row is not None and s.id != row["id"]:
                raise ValueError(f"{project['name']} already has a live Claude session (#{s.id})")
            if s and not follow_up:  # fresh conversation: let the current one finish, then close it
                s.close_when_idle()
                await asyncio.wait([s.task])
                s = None
            if s is None:
                if follow_up and row is None:
                    row = db.one(self.con, "SELECT * FROM claude_sessions WHERE project_id=? ORDER BY id DESC LIMIT 1",
                                 project["id"])
                if follow_up and row is not None:
                    s = Session(self, row, project)
                else:
                    csid = project.get("claude_session_id") if follow_up else None
                    s = Session(self, self._new_row(project, csid), project)
                self.live[project["id"]] = s
            return s.submit(text)

    async def run(self, project: dict, prompt: str, follow_up: bool = False, wait: float | None = None,
                  row: dict | None = None) -> dict:
        """row: continue that specific session (instead of the project's latest), closing another live one first."""
        path = project.get("path")
        if not path or not Path(path).is_dir():
            raise ValueError(f"project folder {path!r} does not exist")
        s = self.live.get(project["id"])
        if row is not None and s and s.id != row["id"]:
            s.close_when_idle()
            await asyncio.wait([s.task])
        fut = await self.submit(project, prompt, follow_up, row)
        if wait is not None:
            done, _ = await asyncio.wait({fut}, timeout=wait)  # does not cancel fut: the turn carries on
            if not done:
                s = self.live.get(project["id"])
                return {"result": "Claude Code is still working on this. The user can watch it live in the Claude "
                                  "tab; results also appear on the project page and in list_claude_sessions.",
                        "session": s.id if s else None, "files_changed": [], "exit_code": None,
                        "timed_out": False, "running": True}
        return fut.result() if fut.done() else await fut

    async def send(self, session_id: int, text: str) -> None:
        row = db.one(self.con, "SELECT * FROM claude_sessions WHERE id=?", session_id)
        if not row:
            raise KeyError(session_id)
        project = db.one(self.con, "SELECT * FROM projects WHERE id=?", row["project_id"])
        await self.submit(project, text, follow_up=True, row=row)

    async def stop(self, session_id: int) -> None:
        s = self.live_by_id(session_id)
        if s:
            await s.stop()

    async def fire_due_resumes(self) -> int:
        due = db.rows(self.con, "SELECT * FROM claude_sessions WHERE status='paused_limit' AND resume_scheduled=1 "
                                "AND resume_at <= ?", db.to_utc(self.now()))
        for row in due:
            lock = self.locks.get(row["project_id"])
            if lock and lock.locked():
                continue  # project is mid-submit: stays scheduled, next tick retries
            db.update(self.con, "claude_sessions", row["id"], resume_scheduled=0)
            project = db.one(self.con, "SELECT * FROM projects WHERE id=?", row["project_id"])
            try:
                await self.submit(project, row["pending_message"] or "Please continue where you left off.",
                                  follow_up=True, row=row)
            except Exception as e:
                db.update(self.con, "claude_sessions", row["id"], resume_scheduled=1, last_error=str(e)[:500])
                log.warning("scheduled resume of session %s skipped: %s", row["id"], e)
        return len(due)

    def list_rows(self) -> list[dict]:
        return db.rows(self.con, "SELECT s.id, s.project_id, p.name AS project, s.status, s.pending_message, "
                                 "s.resume_at, s.resume_estimated, s.resume_scheduled, s.last_error, s.created_at, "
                                 "s.updated_at FROM claude_sessions s JOIN projects p ON p.id = s.project_id "
                                 "ORDER BY s.updated_at DESC, s.id DESC LIMIT 50")

    def usage(self) -> dict:
        return json.loads(db.kv_get(self.con, "claude_usage", "{}") or "{}")

    async def shutdown(self) -> None:
        for s in list(self.live.values()):
            await s.stop()
