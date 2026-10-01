"""tmux service manager: registry, live status/output, start/stop with dependencies and GPU fit,
Claude-made config changes with backups, and pending actions only the user can confirm."""
import asyncio
import contextlib
import difflib
import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

import config
import db
from claude_sessions import _proc_table

log = logging.getLogger("services")
SHELLS = {"bash", "zsh", "sh", "fish", "dash"}
WEBTERM = "webterm-"  # tmux sessions owned by terminal.py; invisible to the services tab and Gemini
FIELDS = ("name", "script_path", "tmux_session", "cwd", "venv", "command", "port", "depends_on", "notes",
          "managed", "project_id")
TRACKED_EXT = {".json", ".yaml", ".yml", ".env", ".sh", ".toml", ".cfg", ".ini", ".txt", ".py"}
MAX_TRACKED = 1_000_000
MAX_SCAN = 20_000
MAX_BACKUP_BYTES = 200_000_000
SKIP_DIRS = {"node_modules", "__pycache__", "chroma_db", "graphify-out", "audio_recordings"}
EDIT_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}
ACTION_KINDS = {"stop", "restart", "send", "register", "start", "start_override_gpu", "http"}
ACTION_TTL = timedelta(minutes=15)

CHANGE_PROMPT = """Make the smallest configuration change so that the service "{name}" runs with: {request}

The service runs in tmux session "{session}", started by {how}. Its folder is {cwd}.
Rules: edit only files inside {cwd} (config files, the start script, env settings). Do not start, stop or
restart anything, do not run the service, and do not call its HTTP endpoints. If nothing needs to change,
say so and change nothing. Finish by listing the files you changed."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_tracked(p: Path, size: int) -> bool:
    n = p.name.lower()  # ".env" has no suffix, so the extension set alone misses it
    return size <= MAX_TRACKED and (p.suffix.lower() in TRACKED_EXT or n == ".env" or n.startswith(".env."))


def _udiff(path: str, old: bytes, new: bytes) -> str:
    a = old.decode(errors="replace").splitlines(keepends=True)
    b = new.decode(errors="replace").splitlines(keepends=True)
    return "".join(difflib.unified_diff(a, b, fromfile=f"a{path}", tofile=f"b{path}"))


class ServiceError(Exception):
    pass


def claim_action(con, aid: int) -> dict:
    """Check a pending action is still valid and mark it running. Returns the row with payload parsed."""
    row = db.one(con, "SELECT * FROM pending_actions WHERE id=?", aid)
    if not row or row["status"] != "pending":
        raise ServiceError("this action is no longer pending")
    if datetime.now(timezone.utc) - db.parse(row["created_at"]) > ACTION_TTL:
        db.update(con, "pending_actions", aid, status="expired", decided_at=db.now())
        raise ServiceError("this action expired; ask again")
    db.update(con, "pending_actions", aid, status="running")
    return {**row, "payload": json.loads(row["payload"] or "{}")}


def existing_action(con, kind: str, service: str, payload: dict) -> int | None:
    """Id of a live pending action identical to this one, so repeating a request doesn't stack duplicates."""
    cutoff = db.to_utc(datetime.now(timezone.utc) - ACTION_TTL)
    want = json.loads(json.dumps(payload))
    for r in db.rows(con, "SELECT id, payload FROM pending_actions WHERE status='pending' AND kind=? AND service=? "
                          "AND created_at >= ? ORDER BY id", kind, service, cutoff):
        if json.loads(r["payload"] or "{}") == want:
            return r["id"]
    return None


def finish_action(con, aid: int, result=None, error: BaseException | None = None) -> None:
    if error is not None:
        db.update(con, "pending_actions", aid, status="failed", result=str(error) or type(error).__name__,
                  decided_at=db.now())
    else:
        db.update(con, "pending_actions", aid, status="done", result=json.dumps(result, default=str)[:4000],
                  decided_at=db.now())


def parse_port(cmd: str) -> int | None:
    for pat in (r"--port[= ]+(\d{2,5})\b", r"--bind[= ]+\S*:(\d{2,5})\b", r"http\.server\s+(\d{2,5})\b",
                r"(?:^|\s)-p\s+(\d{2,5})\b"):
        m = re.search(pat, cmd or "")
        if m:
            return int(m.group(1))
    return None


def parse_script(path: str) -> dict:
    """Read a run_*.sh in the user's pattern: SESSION=..., tmux send-keys "<cd|venv|command>" C-m."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise ServiceError(f"script {path} does not exist")
    text = p.read_text(errors="replace")
    m = re.search(r'^\s*SESSION\s*=\s*["\']?([\w.-]+)', text, re.M)
    session = m.group(1) if m else None
    cwd = venv = command = None
    for line in re.findall(r'send-keys\s+-t\s+\S+\s+"(.*?)"\s+C-m', text):
        line = line.strip()
        if line.startswith("cd "):
            cwd = line[3:].strip().strip("\"'")
        elif re.match(r"^(\.|source)\s+\S*activate\b", line):
            venv = line.split(None, 1)[1].strip()
        else:
            command = line  # the last other line is the service command
    return {"name": session, "tmux_session": session, "script_path": str(p.resolve()),
            "cwd": cwd or str(p.resolve().parent), "venv": venv, "command": command,
            "port": parse_port(command or ""), "ok": bool(session and command)}


def _tree(table: dict, root: int) -> set[int]:
    kids: dict[int, list[int]] = {}
    for pid, (ppid, _) in table.items():
        kids.setdefault(ppid, []).append(pid)
    out, todo = set(), [root]
    while todo:
        p = todo.pop()
        if p not in out:
            out.add(p)
            todo += kids.get(p, [])
    return out


class ServiceManager:
    def __init__(self, con, runner=None, env=None, port_wait: float = 120, stop_grace: float = 10,
                 change_wait: float = 180, backup_root=None):
        self.con, self.runner = con, runner
        self.env = {k: v for k, v in (os.environ if env is None else env).items() if k != "TMUX"}
        self.port_wait, self.stop_grace, self.change_wait = port_wait, stop_grace, change_wait
        self.backup_root = Path(backup_root or config.ROOT / "backups" / "service_changes")
        self._bg: set[asyncio.Task] = set()
        self.con.execute("UPDATE service_changes SET status='failed', explanation='interrupted by restart', "
                         "updated_at=? WHERE status IN ('running','applying')", (db.now(),))

    # ---------- subprocess helpers ----------
    async def _exec(self, *args, timeout: float = 15) -> tuple[int, str]:
        try:
            proc = await asyncio.create_subprocess_exec(
                *args, env=self.env, stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        except FileNotFoundError:
            raise ServiceError(f"{args[0]} is not installed")
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout)
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()
            return 124, f"{args[0]} timed out"
        return proc.returncode, out.decode(errors="replace")

    async def tmux(self, *args, timeout: float = 15) -> tuple[int, str]:
        return await self._exec("tmux", *args, timeout=timeout)

    def _event(self, svc: dict, kind: str, detail: str = "", origin: str = "system") -> None:
        db.insert(self.con, "service_events", service_id=svc["id"], kind=kind,
                  detail=f"[{origin}] {detail}"[:1000], created_at=db.now())

    # ---------- registry ----------
    def get(self, name: str) -> dict:
        r = db.one(self.con, "SELECT * FROM services WHERE name=? OR tmux_session=?", name, name)
        if not r:
            known = ", ".join(x["name"] for x in db.rows(self.con, "SELECT name FROM services ORDER BY name"))
            raise ServiceError(f"No service named '{name}'. Registered: {known or 'none'}.")
        return r

    def register(self, fields: dict) -> dict:
        name = fields.get("name")
        existing = db.one(self.con, "SELECT * FROM services WHERE name=?", name) if name else None
        f = {k: existing[k] for k in FIELDS if existing and existing[k] is not None}
        for k in FIELDS:
            if k in fields:
                if fields[k] in (None, ""):
                    f.pop(k, None)
                else:
                    f[k] = fields[k]
        if not f.get("name"):
            raise ServiceError("a service name is required")
        f.setdefault("tmux_session", f["name"])
        for k in ("name", "tmux_session"):
            if not re.fullmatch(r"[\w.-]+", str(f[k])):
                raise ServiceError(f"{k} '{f[k]}' may only contain letters, digits, '_', '.', '-'")
            if str(f[k]).startswith(WEBTERM):
                raise ServiceError(f"{k} '{f[k]}' is reserved for web terminals")
        # get() looks up by name OR session, so neither may collide with the other's namespace
        other = db.one(self.con, "SELECT name FROM services WHERE id != ? AND (tmux_session IN (?, ?) OR name=?)",
                       existing["id"] if existing else 0, f["tmux_session"], f["name"], f["tmux_session"])
        if other:
            raise ServiceError(f"name/tmux session {f['name']}/{f['tmux_session']} is already used by service "
                               f"{other['name']}")
        if not f.get("script_path") and not f.get("command"):
            raise ServiceError("a service needs a script_path or a command")
        if f.get("script_path") and not Path(f["script_path"]).is_file():
            raise ServiceError(f"script {f['script_path']} does not exist")
        if f.get("port") is not None:
            try:
                f["port"] = int(f["port"])
            except (TypeError, ValueError):
                raise ServiceError(f"invalid port {f['port']!r}")
            if not 1 <= f["port"] <= 65535:
                raise ServiceError(f"invalid port {f['port']}")
        deps = f.get("depends_on") or []
        if isinstance(deps, str):
            try:
                deps = json.loads(deps) if deps.strip().startswith("[") else deps.split(",")
            except ValueError:
                raise ServiceError("depends_on is not valid JSON")
        if not isinstance(deps, list):
            raise ServiceError("depends_on must be a list")
        deps = [str(d).strip() for d in deps if str(d).strip()]
        for d in deps:
            if d == f["name"]:
                raise ServiceError("a service cannot depend on itself")
            if not db.one(self.con, "SELECT 1 FROM services WHERE name=?", d):
                raise ServiceError(f"dependency '{d}' is not a registered service")
        f["depends_on"] = json.dumps(deps)
        if existing:
            cleared = {k: None for k in FIELDS if k not in f}
            db.update(self.con, "services", existing["id"], updated_at=db.now(), **{**cleared, **f})
        else:
            db.insert(self.con, "services", created_at=db.now(), updated_at=db.now(), **f)
        return self.get(f["name"])

    def unregister(self, name: str) -> None:
        svc = self.get(name)
        self.con.execute("DELETE FROM services WHERE id=?", (svc["id"],))
        self.con.execute("DELETE FROM kv WHERE key=?", (f"svc_up:{svc['name']}",))

    def _managed(self, name: str) -> dict:
        svc = self.get(name)
        if not svc["managed"]:
            raise ServiceError(f"{svc['name']} is view-only")
        return svc

    def project_for(self, svc: dict) -> dict:
        cwd = str(Path(svc["cwd"]).resolve())
        p = db.one(self.con, "SELECT * FROM projects WHERE path=?", cwd)
        if not p:
            name = f"{svc['name']}-service"
            ex = db.one(self.con, "SELECT id FROM projects WHERE name=?", name)
            if ex:
                db.update(self.con, "projects", ex["id"], path=cwd)
                pid = ex["id"]
            else:
                pid = db.insert(self.con, "projects", name=name, path=cwd, goal=f"Service {svc['name']}",
                                created_at=db.now(), last_activity_at=db.now())
            p = db.one(self.con, "SELECT * FROM projects WHERE id=?", pid)
        return p

    def import_candidates(self) -> list[dict]:
        known = {s["script_path"] for s in db.rows(self.con, "SELECT script_path FROM services")}
        out = []
        roots = sorted({r["path"] for r in db.rows(self.con, "SELECT path FROM projects WHERE path IS NOT NULL AND path != ''")})
        for root in roots:
            base = Path(root)
            if not base.is_dir():
                continue
            for script in sorted(base.glob("run_*.sh")) + sorted(base.glob("*/run_*.sh")):
                sp = str(script.resolve())
                if sp in known:
                    continue
                with contextlib.suppress(OSError, ServiceError):
                    out.append(parse_script(sp))
                    known.add(sp)
        return out

    # ---------- observation ----------
    async def sessions(self) -> dict[str, dict]:
        code, out = await self.tmux("list-panes", "-a", "-F", "#{session_name}|#{pane_pid}|#{pane_current_command}")
        if code != 0:
            if "no server running" in out or "error connecting" in out:
                return {}
            raise ServiceError(f"tmux failed: {out.strip()}")
        res: dict[str, dict] = {}
        for line in out.splitlines():
            parts = line.split("|")
            if len(parts) == 3 and parts[1].isdigit() and not parts[0].startswith(WEBTERM):  # web terminals are private
                res.setdefault(parts[0], {"pane_pid": int(parts[1]), "pane_command": parts[2]})
        return res

    async def listening_ports(self) -> set[int] | None:
        try:
            code, out = await self._exec("ss", "-ltnH")
        except ServiceError:
            return None
        if code != 0:
            return None
        ports = set()
        for line in out.splitlines():
            cols = line.split()
            if len(cols) >= 4:
                m = re.search(r":(\d+)$", cols[3])
                if m:
                    ports.add(int(m.group(1)))
        return ports

    async def port_up(self, port: int, ports: set[int] | None = None) -> bool:
        if ports is None:
            ports = await self.listening_ports()
        if ports is not None:
            return port in ports
        try:
            _, w = await asyncio.wait_for(asyncio.open_connection("127.0.0.1", port), 1)
            w.close()
            return True
        except (OSError, asyncio.TimeoutError):
            return False

    async def gpu(self) -> dict | None:
        try:
            c1, tot = await self._exec("nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits")
            c2, apps = await self._exec("nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits")
        except ServiceError:
            return None
        if c1 != 0:
            return None
        used = total = 0
        for line in tot.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 2 and all(x.isdigit() for x in parts):
                used, total = used + int(parts[0]), total + int(parts[1])
        procs: dict[int, int] = {}
        for line in apps.splitlines() if c2 == 0 else []:
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 2 and all(x.isdigit() for x in parts):
                procs[int(parts[0])] = procs.get(int(parts[0]), 0) + int(parts[1])
        return {"used_mb": used, "total_mb": total, "procs": procs} if total else None

    def _session_name(self, name: str) -> str:
        r = db.one(self.con, "SELECT tmux_session FROM services WHERE name=? OR tmux_session=?", name, name)
        return r["tmux_session"] if r else name

    async def output(self, name: str, lines: int = 300) -> str:
        sess = self._session_name(name)
        if sess.startswith(WEBTERM):
            raise ServiceError(f"tmux session '{sess}' is not running")
        code, out = await self.tmux("capture-pane", "-p", "-J", "-t", f"={sess}:", "-S", f"-{int(lines)}")
        if code != 0:
            raise ServiceError(f"tmux session '{sess}' is not running")
        return out.rstrip("\n")

    async def _last_line(self, sess: str) -> str | None:
        with contextlib.suppress(ServiceError):
            lines = [x for x in (await self.output(sess, 30)).splitlines() if x.strip()]
            return lines[-1][:200] if lines else None
        return None

    async def status(self) -> dict:
        error = None
        try:
            live = await self.sessions()
        except ServiceError as e:
            live, error = {}, str(e)
        ports = await self.listening_ports()
        g = await self.gpu()
        table = _proc_table() if g else {}
        reg = {s["tmux_session"]: s for s in db.rows(self.con, "SELECT * FROM services ORDER BY name")}
        items = []
        for sess in list(reg) + sorted(set(live) - set(reg)):
            s, pane = reg.get(sess), live.get(sess)
            port = s["port"] if s else None
            item = {"name": s["name"] if s else sess, "tmux_session": sess, "registered": s is not None,
                    "managed": bool(s and s["managed"]), "alive": pane is not None,
                    "pane_command": pane["pane_command"] if pane else None, "port": port,
                    "port_up": (await self.port_up(port, ports)) if port else None,
                    "should_run": bool(s and s["should_run"]), "gpu_mb": None, "last_line": None,
                    "depends_on": json.loads(s["depends_on"] or "[]") if s else [],
                    "gpu_mem_gb": s["gpu_mem_gb"] if s else None, "script_path": s["script_path"] if s else None,
                    "cwd": s["cwd"] if s else None, "command": s["command"] if s else None,
                    "notes": s["notes"] if s else None}
            if pane and g:
                item["gpu_mb"] = sum(g["procs"].get(p, 0) for p in _tree(table, pane["pane_pid"]))
            if pane:
                item["last_line"] = await self._last_line(sess)
            items.append(item)
        gpu = {"used_mb": g["used_mb"], "total_mb": g["total_mb"]} if g else None
        return {"services": items, "gpu": gpu, "error": error}

    # ---------- control ----------
    def _refuse_self(self, svc: dict, live: dict) -> None:
        pane = live.get(svc["tmux_session"])
        if pane and os.getpid() in _tree(_proc_table(), pane["pane_pid"]):
            raise ServiceError("that tmux session runs this assistant")

    async def send(self, name: str, text: str | None = None, ctrl_c: bool = False, origin: str = "ui") -> dict:
        svc = self._managed(name)
        target = f"={svc['tmux_session']}:"
        live = await self.sessions()
        if svc["tmux_session"] not in live:
            raise ServiceError(f"{svc['name']} is not running")
        if ctrl_c:
            self._refuse_self(svc, live)
        if ctrl_c:
            await self.tmux("send-keys", "-t", target, "C-c")
        if text:
            await self.tmux("send-keys", "-t", target, "-l", "--", text)
            await self.tmux("send-keys", "-t", target, "Enter")
        self._event(svc, "send", ("^C " if ctrl_c else "") + (text or ""), origin)
        return {"ok": True}

    async def _wait_port(self, port: int, timeout: float, sess: str | None = None) -> bool:
        t0 = time.monotonic()
        deadline = t0 + timeout
        while time.monotonic() < deadline:
            if await self.port_up(port):
                return True
            if sess and time.monotonic() - t0 > 3 and (await self.sessions()).get(sess, {}).get("pane_command") in SHELLS:
                break  # the service command has exited
            await asyncio.sleep(1)
        return await self.port_up(port)

    async def _launch(self, svc: dict) -> None:
        if svc["script_path"]:
            script = Path(svc["script_path"])
            proc = await asyncio.create_subprocess_exec(
                "bash", str(script), cwd=str(script.parent), env=self.env, start_new_session=True,
                stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            try:  # the trailing `tmux attach` fails without a terminal; that's expected
                await asyncio.wait_for(proc.communicate(), 30)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                await proc.wait()
            return
        sess, target = svc["tmux_session"], f"={svc['tmux_session']}:"
        code, out = await self.tmux("new-session", "-d", "-s", sess, "-c", svc["cwd"] or str(Path.home()))
        if code != 0:
            raise ServiceError(f"tmux new-session failed: {out.strip()}")
        if svc["venv"]:
            act = svc["venv"] if svc["venv"].endswith("activate") else f"{svc['venv'].rstrip('/')}/bin/activate"
            await self.tmux("send-keys", "-t", target, "-l", "--", f". {act}")
            await self.tmux("send-keys", "-t", target, "Enter")
        await self.tmux("send-keys", "-t", target, "-l", "--", svc["command"])
        await self.tmux("send-keys", "-t", target, "Enter")

    async def _gpu_consumers(self, g: dict) -> list[dict]:
        live = await self.sessions()
        table = _proc_table()
        out = []
        for sess, pane in live.items():
            mb = sum(g["procs"].get(p, 0) for p in _tree(table, pane["pane_pid"]))
            if mb:
                out.append({"tmux_session": sess, "gpu_mb": mb})
        return sorted(out, key=lambda x: -x["gpu_mb"])

    def _pending_refusal(self, svc: dict) -> dict | None:
        """Apply/undo restart while the change is 'applying'/'undone', so only running/proposed block."""
        pend = db.one(self.con, "SELECT id, status FROM service_changes WHERE service_id=? "
                                "AND status IN ('running','proposed')", svc["id"])
        return None if not pend else {
            "ok": False, "service": svc["name"], "started": [],
            "error": f"A config change for {svc['name']} is waiting (#{pend['id']}, {pend['status']}): "
                     "tap Apply or Undo first."}

    async def start(self, name: str, override_gpu: bool = False, _path: frozenset = frozenset()) -> dict:
        svc = self._managed(name)
        if svc["name"] in _path:
            return {"ok": False, "service": svc["name"], "started": [], "error": f"dependency cycle at {svc['name']}"}
        if not _path and (refusal := self._pending_refusal(svc)):  # top-level only, not dependency starts
            return refusal
        path = _path | {svc["name"]}
        started: list[str] = []
        for dep in json.loads(svc["depends_on"] or "[]"):
            r = await self.start(dep, override_gpu, path)
            if not r["ok"]:
                return {**r, "ok": False, "service": svc["name"], "error": f"dependency {dep} failed: {r.get('error')}"}
            started += r["started"]
        sess = svc["tmux_session"]
        if sess in await self.sessions():
            if not svc["port"] or await self.port_up(svc["port"]):
                db.update(self.con, "services", svc["id"], should_run=1)
                return {"ok": True, "service": svc["name"], "started": started, "already_running": True}
            return {"ok": False, "service": svc["name"], "started": started,
                    "error": f"tmux session {sess} exists but port {svc['port']} is not listening; restart it",
                    "tail": await self.output(sess, 50)}
        if svc["port"] and await self.port_up(svc["port"]):
            return {"ok": False, "service": svc["name"], "started": started,
                    "error": f"port {svc['port']} is already in use by another process"}
        if svc["gpu_mem_gb"] and not override_gpu:
            g = await self.gpu()
            if g and g["total_mb"] - g["used_mb"] < svc["gpu_mem_gb"] * 1024:
                free = (g["total_mb"] - g["used_mb"]) / 1024
                return {"ok": False, "service": svc["name"], "started": started, "gpu_blocked": True,
                        "error": f"not enough GPU memory: {svc['name']} needs ~{svc['gpu_mem_gb']:.1f} GB, "
                                 f"{free:.1f} GB free", "consumers": await self._gpu_consumers(g)}
        await self._launch(svc)
        if sess not in await self.sessions():
            return {"ok": False, "service": svc["name"], "started": started, "error": "the tmux session did not start"}
        if svc["port"] and not await self._wait_port(svc["port"], self.port_wait, sess):
            return {"ok": False, "service": svc["name"], "started": started,
                    "error": f"port {svc['port']} not listening after {self.port_wait:g}s",
                    "tail": await self.output(sess, 50)}
        db.update(self.con, "services", svc["id"], should_run=1, updated_at=db.now())
        self._event(svc, "start", "override_gpu" if override_gpu else "")
        with contextlib.suppress(Exception):
            await self.learn_gpu(svc)
        return {"ok": True, "service": svc["name"], "started": started + [svc["name"]]}

    async def stop(self, name: str) -> dict:
        svc = self._managed(name)
        sess = svc["tmux_session"]
        live = await self.sessions()
        self._refuse_self(svc, live)
        db.update(self.con, "services", svc["id"], should_run=0, updated_at=db.now())
        if sess not in live:
            return {"ok": True, "service": svc["name"], "already_stopped": True}
        await self.tmux("send-keys", "-t", f"={sess}:", "C-c")
        deadline = time.monotonic() + self.stop_grace
        while time.monotonic() < deadline:
            await asyncio.sleep(0.5)
            if svc["port"]:
                if not await self.port_up(svc["port"]):
                    break
            elif (await self.sessions()).get(sess, {}).get("pane_command") in SHELLS:
                break
        await self.tmux("kill-session", "-t", f"={sess}")
        self._event(svc, "stop")
        return {"ok": True, "service": svc["name"]}

    async def restart(self, name: str) -> dict:
        if refusal := self._pending_refusal(self._managed(name)):  # refuse before stopping, never leave it down
            return refusal
        await self.stop(name)
        return await self.start(name)

    # ---------- config changes ----------
    def _walk(self, root: Path):
        for dirpath, dirnames, filenames in os.walk(root):
            depth = len(Path(dirpath).relative_to(root).parts)
            dirnames[:] = [] if depth >= 4 else [
                d for d in dirnames
                if not (d in SKIP_DIRS or d.startswith(".") or d.lower().startswith("model")
                        or d.lower().startswith(("venv", "data")) or d.lower().endswith("env")
                        or d == "site-packages" or (Path(dirpath) / d / "pyvenv.cfg").exists())]
            for f in filenames:
                yield Path(dirpath) / f

    def _scan(self, root: Path) -> tuple[dict[str, list[int]], bool]:
        out: dict[str, list[int]] = {}
        for i, p in enumerate(self._walk(root)):
            if i >= MAX_SCAN:
                return out, True
            with contextlib.suppress(OSError):
                st = p.stat()
                out[str(p)] = [st.st_size, st.st_mtime_ns]
        return out, False

    def _backup(self, files: list[str], bdir: Path) -> dict:
        bdir.mkdir(parents=True, exist_ok=True)
        tracked, total = {}, 0
        for i, path in enumerate(files):
            try:
                data = Path(path).read_bytes()
                total += len(data)
                if total > MAX_BACKUP_BYTES:
                    raise ServiceError("too much to back up")
                (bdir / f"{i}.bak").write_bytes(data)
            except OSError as e:
                raise ServiceError(f"cannot back up {path}: {e}")
            tracked[path] = {"bak": f"{i}.bak", "sha": _sha(data)}
        return tracked

    def _edited_paths(self, session_id, since: str) -> list[str]:
        if not session_id:
            return []
        paths = []
        for ev in db.rows(self.con, "SELECT payload FROM claude_events WHERE session_id=? AND kind='assistant' "
                                    "AND created_at >= ?", session_id, since):
            with contextlib.suppress(ValueError, AttributeError, TypeError):
                for c in json.loads(ev["payload"]).get("message", {}).get("content", []):
                    if c.get("type") == "tool_use" and c.get("name") in EDIT_TOOLS:
                        fp = (c.get("input") or {}).get("file_path") or (c.get("input") or {}).get("notebook_path")
                        if fp:
                            paths.append(fp)
        return paths

    def _name_of(self, sid: int) -> str:
        return (db.one(self.con, "SELECT name FROM services WHERE id=?", sid) or {}).get("name", "a service")

    def change(self, cid: int) -> dict:
        r = db.one(self.con, "SELECT * FROM service_changes WHERE id=?", cid)
        if not r:
            raise ServiceError(f"change {cid} not found")
        r.pop("manifest", None)
        r["result"] = json.loads(r["result"] or "{}")
        r["service"] = (db.one(self.con, "SELECT name FROM services WHERE id=?", r["service_id"]) or {}).get("name")
        return r

    def changes_for(self, name: str) -> list[dict]:
        svc = self.get(name)
        return [self.change(r["id"]) for r in
                db.rows(self.con, "SELECT id FROM service_changes WHERE service_id=? ORDER BY id DESC LIMIT 20", svc["id"])]

    async def change_config(self, name: str, request: str) -> dict:
        if self.runner is None:
            raise ServiceError("Claude sessions are not available")
        svc = self._managed(name)
        if not svc["cwd"] or not Path(svc["cwd"]).is_dir():
            raise ServiceError(f"service folder {svc['cwd']!r} does not exist")
        root = Path(svc["cwd"]).resolve()
        same = [s["id"] for s in db.rows(self.con, "SELECT id, cwd FROM services WHERE cwd IS NOT NULL")
                if Path(s["cwd"]).resolve() == root]  # services sharing the folder share the backup/diff
        busy = db.one(self.con, f"SELECT id, status, service_id FROM service_changes WHERE status IN "
                                f"('running','applying','proposed') AND service_id IN ({','.join('?' * len(same))})",
                      *same) if same else None
        if busy:
            other = self._name_of(busy["service_id"])
            raise ServiceError(f"{other} already has a change in progress or waiting for Apply/Undo "
                               f"(#{busy['id']}, {busy['status']}) in this folder")
        project = self.project_for(svc)
        started = db.now()
        cid = db.insert(self.con, "service_changes", service_id=svc["id"], request=request, status="running",
                        created_at=started, updated_at=started)
        bdir = self.backup_root / str(cid)
        try:
            scan, truncated = await asyncio.to_thread(self._scan, root)
            files = [p for p, (size, _) in scan.items() if _is_tracked(Path(p), size)]
            if svc["script_path"] and svc["script_path"] not in files and Path(svc["script_path"]).is_file():
                files.append(svc["script_path"])
            tracked = await asyncio.to_thread(self._backup, files, bdir)
            manifest = {"root": str(root), "scan": scan, "truncated": truncated, "tracked": tracked, "started": started}
            db.update(self.con, "service_changes", cid, backup_dir=str(bdir), manifest=json.dumps(manifest))
        except Exception as e:
            log.exception("change %s: backup failed", cid)
            db.update(self.con, "service_changes", cid, status="failed", explanation=f"backup failed: {e}"[:4000],
                      updated_at=db.now())
            return self.change(cid)
        how = f"the script {svc['script_path']}" if svc["script_path"] else f"the command `{svc['command']}`"
        prompt = CHANGE_PROMPT.format(name=svc["name"], request=request, session=svc["tmux_session"], how=how, cwd=root)
        task = asyncio.create_task(self._finish_change(cid, project, prompt))
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)
        await asyncio.wait({task}, timeout=self.change_wait)  # doesn't cancel: a slow change keeps running
        return self.change(cid)

    async def _finish_change(self, cid: int, project: dict, prompt: str) -> None:
        try:
            res = await self.runner.run(project, prompt, follow_up=False)
            explanation = str(res.get("result", ""))[:4000]
            row = db.one(self.con, "SELECT manifest, backup_dir FROM service_changes WHERE id=?", cid)
            manifest = json.loads(row["manifest"])
            edited = self._edited_paths(res.get("session"), manifest["started"])
            result = await asyncio.to_thread(self._compare, manifest, row["backup_dir"], edited)
            touched = result["changed"] or result["created"] or result["deleted"] or result["other"]
            # a failed run keeps status failed but still records what it touched, so undo can revert it
            status = "failed" if res.get("final") or res.get("exit_code") == 1 else "proposed" if touched or result["flags"] else "no_change"
            db.update(self.con, "service_changes", cid, status=status, diff=result.pop("diff"),
                      flagged=int(bool(result["flags"])), explanation=explanation,
                      claude_session=res.get("session"), result=json.dumps(result), updated_at=db.now())
        except Exception as e:
            log.exception("change %s failed", cid)
            db.update(self.con, "service_changes", cid, status="failed", explanation=str(e)[:4000], updated_at=db.now())

    def _compare(self, manifest: dict, backup_dir: str, edited: list[str]) -> dict:
        root, before, tracked = Path(manifest["root"]), manifest["scan"], manifest["tracked"]
        after, truncated = self._scan(root)
        changed, created, deleted, other, flags, diffs, also = [], [], [], [], [], [], []
        after_sha: dict[str, str | None] = {}  # state right after Claude's change; undo skips files edited since
        for path, info in tracked.items():
            old = (Path(backup_dir) / info["bak"]).read_bytes()
            p = Path(path)
            if not p.exists():
                deleted.append(path)
                after_sha[path] = None
                diffs.append(_udiff(path, old, b""))
                continue
            new = p.read_bytes()
            if _sha(new) != info["sha"]:
                changed.append(path)
                after_sha[path] = _sha(new)
                diffs.append(_udiff(path, old, new))
        for path, meta in after.items():
            if path in tracked:
                continue
            if path not in before:
                if _is_tracked(Path(path), meta[0]):
                    data = Path(path).read_bytes()
                    created.append(path)
                    after_sha[path] = _sha(data)
                    diffs.append(_udiff(path, b"", data))
                else:  # untracked changes are usually the running service (logs, data): informational only
                    also.append({"path": path, "kind": "created"})
            elif before[path] != meta:
                also.append({"path": path, "kind": "modified"})
        also += [{"path": p, "kind": "deleted"} for p in before if p not in after and p not in tracked]
        if truncated or manifest.get("truncated"):
            flags.append("folder too large to scan fully")
        for fp in edited:  # what Claude itself edited but we could not back up: flagged and not restorable
            p = Path(fp)
            resolved = (p if p.is_absolute() else root / p).resolve()
            if str(resolved) in tracked or fp in tracked or str(resolved) in created:
                continue
            other.append(str(resolved))
            flags.append(f"edited outside the service folder: {fp}" if not resolved.is_relative_to(root)
                         else f"edited a file that was not backed up: {fp}")
        return {"changed": changed, "created": created, "deleted": deleted, "other": other, "flags": flags,
                "also_changed": also[:50], "after_sha": after_sha, "diff": "".join(diffs)[:60_000]}

    async def apply(self, cid: int) -> dict:
        row = self.change(cid)
        if row["status"] != "proposed":
            raise ServiceError(f"change {cid} is {row['status']}, not proposed")
        if row["flagged"]:
            raise ServiceError("this change is flagged (it touched files outside the tracked set); review it — it can only be undone")
        raw = db.one(self.con, "SELECT manifest, backup_dir FROM service_changes WHERE id=?", cid)
        now = await asyncio.to_thread(self._compare, json.loads(raw["manifest"]), raw["backup_dir"], [])
        # only the tracked set matters; log/data writes by the running service (also_changed) are ignored
        norm = lambda v: sorted(v) if isinstance(v, list) else v  # noqa: E731
        if now["flags"] or any(norm(now[k]) != norm(row["result"].get(k))
                               for k in ("changed", "created", "deleted", "after_sha")):
            raise ServiceError("Files changed since this diff was made — review it again (undo or ask for a new change)")
        svc = db.one(self.con, "SELECT name FROM services WHERE id=?", row["service_id"])
        db.update(self.con, "service_changes", cid, status="applying", updated_at=db.now())
        try:
            restart = await self.restart(svc["name"])
        except BaseException:
            db.update(self.con, "service_changes", cid, status="proposed", updated_at=db.now())
            raise
        db.update(self.con, "service_changes", cid, status="applied", updated_at=db.now(),
                  result=json.dumps({**row["result"], "apply_restart": restart}, default=str))
        return {"change": cid, "restart": restart}

    async def undo(self, cid: int) -> dict:
        r = db.one(self.con, "SELECT * FROM service_changes WHERE id=?", cid)
        if not r:
            raise ServiceError(f"change {cid} not found")
        if r["status"] not in ("proposed", "applied", "no_change", "failed", "undone_partial"):
            raise ServiceError(f"change {cid} is {r['status']}; nothing to undo")
        manifest, result = json.loads(r["manifest"] or "{}"), json.loads(r["result"] or "{}")
        if "changed" not in result and "root" in manifest:  # interrupted or raised: see what it left behind
            result = await asyncio.to_thread(self._compare, manifest, r["backup_dir"], [])
            db.update(self.con, "service_changes", cid, diff=result.pop("diff"), result=json.dumps(result))
        restored, conflicts = [], []
        tracked, after_sha = manifest.get("tracked", {}), result.get("after_sha", {})

        def cur(p: Path) -> str | None:
            return _sha(p.read_bytes()) if p.exists() else None

        for path in result.get("changed", []) + result.get("deleted", []):
            if path in tracked:  # only what Claude touched; other files may have been written by the service since
                p, now = Path(path), cur(Path(path))
                if now == tracked[path]["sha"]:
                    continue  # already back to the original (an earlier partial undo)
                if now != after_sha.get(path):
                    conflicts.append(path)  # edited after Claude's change: keep the user's version
                    continue
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes((Path(r["backup_dir"]) / tracked[path]["bak"]).read_bytes())
                restored.append(path)
        for path in result.get("created", []):
            now = cur(Path(path))
            if now is None:
                continue
            if now != after_sha.get(path):
                conflicts.append(path)
                continue
            Path(path).unlink()
            restored.append(path)
        db.update(self.con, "service_changes", cid, status="undone_partial" if conflicts else "undone",
                  updated_at=db.now())
        restart = None
        if r["status"] == "applied":
            svc = db.one(self.con, "SELECT name FROM services WHERE id=?", r["service_id"])
            restart = await self.restart(svc["name"])
        return {"change": cid, "restored": restored, "conflicts": conflicts,
                "not_restored": result.get("other", []), "restart": restart}

    async def http(self, name: str, method: str, path: str, body=None) -> dict:
        svc = self._managed(name)
        if not svc["port"]:
            raise ServiceError(f"service {svc['name']} has no port registered")
        url = f"http://127.0.0.1:{svc['port']}{path}"
        try:
            async with httpx.AsyncClient(timeout=30) as c:
                r = await c.request(method, url, json=body)
        except httpx.HTTPError as e:
            return {"error": f"could not connect to {url}: {e}"}
        return {"status": r.status_code, "body": r.text[:4000], "url": url}

    # ---------- pending actions ----------
    def create_action(self, kind: str, service: str, payload: dict | None = None, created_by: str = "gemini") -> dict:
        if kind not in ACTION_KINDS:
            raise ServiceError(f"unknown action {kind}")
        payload = payload or {}
        if kind != "register":
            self._managed(service)
        if kind == "send":
            text = payload.get("text") or ""
            if not text and not payload.get("ctrl_c"):
                raise ServiceError("send needs text or ctrl_c")
            if len(text) > 2000:
                raise ServiceError("text is too long to send (max 2000 characters)")
            if "\n" in text or "\r" in text:
                raise ServiceError("send one line at a time")
        body = payload.get("body")
        summary = {"stop": f"Stop {service}", "restart": f"Restart {service}",
                   "send": f"Send to {service}: {payload.get('text') or '^C'}",
                   "register": f"Register service {service}: {payload.get('script_path') or payload.get('command')}"
                               + (" (replaces the existing service)"
                                  if db.one(self.con, "SELECT 1 FROM services WHERE name=?", service) else ""),
                   "start": f"Start {service}",
                   "start_override_gpu": f"Start {service} despite low GPU memory",
                   "http": f"{payload.get('method')} {payload.get('path')} on {service}"
                           + (f" with {json.dumps(body)[:200]}" if body is not None else "")}[kind]
        aid = existing_action(self.con, kind, service, payload)
        if aid is None:
            aid = db.insert(self.con, "pending_actions", kind=kind, service=service, payload=json.dumps(payload),
                            created_by=created_by, status="pending", created_at=db.now())
        return {"needs_confirmation": True, "action_id": aid, "kind": kind, "service": service, "summary": summary}

    def pending_actions(self) -> list[dict]:
        cutoff = db.to_utc(datetime.now(timezone.utc) - ACTION_TTL)
        return db.rows(self.con, "SELECT * FROM pending_actions WHERE status='pending' AND created_at >= ? "
                                 "ORDER BY id DESC", cutoff)

    def cancel_action(self, aid: int) -> None:
        self.con.execute("UPDATE pending_actions SET status='cancelled', decided_at=? WHERE id=? AND status='pending'",
                         (db.now(), aid))

    async def confirm_action(self, aid: int) -> dict:
        row = claim_action(self.con, aid)
        payload = row["payload"]
        try:
            if row["kind"] == "stop":
                res = await self.stop(row["service"])
            elif row["kind"] == "restart":
                res = await self.restart(row["service"])
            elif row["kind"] == "send":
                res = await self.send(row["service"], payload.get("text"), bool(payload.get("ctrl_c")), origin="gemini")
            elif row["kind"] == "register":
                res = {"ok": True, "registered": self.register(payload)["name"]}
            elif row["kind"] == "start":
                res = await self.start(row["service"])
            elif row["kind"] == "start_override_gpu":
                res = await self.start(row["service"], override_gpu=True)
            elif row["kind"] == "http":
                res = await self.http(row["service"], payload.get("method"), payload.get("path"), payload.get("body"))
                if "error" in res:  # a call that never reached the service is a failure, not "done"
                    raise ServiceError(res["error"])
            else:
                raise ServiceError(f"{row['kind']} is not a service action")
        except Exception as e:
            finish_action(self.con, aid, error=e)
            raise
        finish_action(self.con, aid, result=res)
        return res

    # ---------- learning & health ----------
    async def learn_gpu(self, svc: dict) -> float | None:
        g = await self.gpu()
        pane = (await self.sessions()).get(svc["tmux_session"])
        if not g or not pane:
            return None
        gb = round(sum(g["procs"].get(p, 0) for p in _tree(_proc_table(), pane["pane_pid"])) / 1024, 2)
        if gb > (svc["gpu_mem_gb"] or 0):
            db.update(self.con, "services", svc["id"], gpu_mem_gb=gb)
        return gb

    async def health_tick(self) -> list[dict]:
        try:
            st = await self.status()
        except ServiceError:
            return []
        changes = []
        for it in st["services"]:
            if not it["registered"]:
                continue
            up = it["alive"] and it["port_up"] is not False
            key = f"svc_up:{it['name']}"
            prev, cur = db.kv_get(self.con, key), "1" if up else "0"
            if prev is not None and prev != cur:
                svc = self.get(it["name"])
                if up or svc["should_run"]:
                    self._event(svc, "up" if up else "down", it["last_line"] or "")
                    changes.append({"name": it["name"], "up": up})
            db.kv_set(self.con, key, cur)
            if up and it["gpu_mb"] and it["gpu_mb"] / 1024 > (it["gpu_mem_gb"] or 0):
                db.update(self.con, "services", self.get(it["name"])["id"], gpu_mem_gb=round(it["gpu_mb"] / 1024, 2))
        return changes

    def down_services(self) -> list[dict]:
        return [s for s in db.rows(self.con, "SELECT name, tmux_session, port FROM services WHERE should_run=1")
                if db.kv_get(self.con, f"svc_up:{s['name']}") == "0"]
