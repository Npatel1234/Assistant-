"""FastAPI app: password auth, JSON API, SSE chat, background scheduler. Run: uv run main.py"""
import asyncio
import contextlib
import hmac
import json
import logging
import logging.handlers
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from urllib.parse import urlparse

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Request, WebSocket
from starlette.requests import HTTPConnection
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
from claude_sessions import SessionManager
from llm import LLM, AllModelsExhausted, pack, unpack
from services import ACTION_TTL, ServiceError, ServiceManager, parse_script
from terminal import TerminalError, TerminalManager, valid_name
from workspace import SUITE_BRANCH, RepoManager

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


class SendIn(BaseModel):
    text: str


class ResumeIn(BaseModel):
    message: str | None = None
    at: str | None = None


def pick(body: dict, allowed: set) -> dict:
    return {k: v for k, v in body.items() if k in allowed and v is not None}


def create_app(con, llm, runner, start_jobs: bool = True, services=None, repos=None,
               terminals=None) -> FastAPI:
    signer = URLSafeTimedSerializer(config.SESSION_SECRET, salt="login")
    failures: dict[str, deque] = defaultdict(deque)
    unlocked: dict[str, float] = {}  # login cookie -> last terminal activity; in memory, restart = locked

    @asynccontextmanager
    async def lifespan(app):
        job = asyncio.create_task(jobs.scheduler(con, llm, runner, services)) if start_jobs else None
        yield
        if job:
            job.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await job
        if runner is not None and hasattr(runner, "shutdown"):
            await runner.shutdown()

    app = FastAPI(lifespan=lifespan)

    def ctx(cid=None) -> tools.Ctx:
        return tools.Ctx(con, llm, runner, cid, services, repos)

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

    def login_token(conn: HTTPConnection) -> str | None:
        """The session cookie if it is validly signed (works for HTTP and WebSocket), else None."""
        token = conn.cookies.get(COOKIE)
        if not token:
            return None
        try:
            signer.loads(token, max_age=MAX_AGE)
            return token
        except BadSignature:
            return None

    def logged_in(request: Request) -> bool:
        return login_token(request) is not None

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

    def check_password(password: str, request: Request) -> None:
        """Constant-time compare with per-IP failure limiting; shared by login and terminal unlock."""
        ip = request.client.host if request.client else "?"
        q, now = failures[ip], time.time()
        while q and q[0] < now - 900:
            q.popleft()
        if len(q) >= 5:
            raise HTTPException(429, "Too many attempts. Try again in 15 minutes.")
        if not hmac.compare_digest(password.encode(), config.APP_PASSWORD.encode()):
            q.append(now)
            raise HTTPException(401, "Wrong password")
        q.clear()

    @app.post("/login")
    async def login(body: LoginIn, request: Request):
        check_password(body.password, request)
        resp = JSONResponse({"ok": True})
        resp.set_cookie(COOKIE, signer.dumps("me"), max_age=MAX_AGE, httponly=True, samesite="lax",
                        secure=request.url.scheme == "https")
        return resp

    @app.post("/logout")
    async def logout(request: Request):
        unlocked.pop(login_token(request), None)  # logging out also locks the terminal
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
                "dead": sorted(llm.dead), "sync_error": db.kv_get(con, "sync_error", ""),
                "clock_offset_s": float(db.kv_get(con, "clock_offset_s", "0") or 0)}

    @app.post("/api/sync")
    async def sync_now():
        return await jobs.sync_claude_history(con, llm, config.CLAUDE_HISTORY_DIR, config.SYNC_MAX_CALLS)

    # ---------- claude sessions ----------

    def sessions_or_503():
        if runner is None or not hasattr(runner, "list_rows"):
            raise HTTPException(503, "Claude sessions are not available")
        return runner

    @app.get("/api/claude/sessions")
    async def claude_sessions():
        return sessions_or_503().list_rows()

    @app.get("/api/claude/usage")
    async def claude_usage():
        return sessions_or_503().usage()

    @app.get("/api/claude/sessions/{sid}/events")
    async def claude_events(sid: int, after: int = 0):
        get_or_404("claude_sessions", sid)
        return db.rows(con, "SELECT seq, kind, payload FROM claude_events WHERE session_id=? AND seq>? ORDER BY seq",
                       sid, after)

    @app.get("/api/claude/sessions/{sid}/stream")
    async def claude_stream(sid: int, after: int = 0):
        get_or_404("claude_sessions", sid)
        live = sessions_or_503().live_by_id(sid)
        q = live.subscribe() if live else None  # subscribe before replay so nothing is missed

        async def gen():
            last = after
            try:
                for ev in db.rows(con, "SELECT seq, kind, payload FROM claude_events WHERE session_id=? AND seq>? "
                                       "ORDER BY seq", sid, after):
                    last = ev["seq"]
                    yield f"data: {json.dumps(ev)}\n\n"
                while q is not None:
                    try:
                        ev = await asyncio.wait_for(q.get(), 15)
                    except asyncio.TimeoutError:
                        yield ": ping\n\n"
                        continue
                    if ev is None:
                        break
                    if ev["seq"] > last:
                        last = ev["seq"]
                        yield f"data: {json.dumps(ev)}\n\n"
                yield "event: end\ndata: {}\n\n"
            finally:
                if live is not None and q is not None:
                    live.watchers.discard(q)

        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.post("/api/claude/sessions/{sid}/send")
    async def claude_send(sid: int, body: SendIn):
        text = body.text.strip()
        if not text:
            raise HTTPException(400, "empty message")
        get_or_404("claude_sessions", sid)
        try:
            await sessions_or_503().send(sid, text)
        except ValueError as e:
            raise HTTPException(409, str(e))
        return {"ok": True}

    @app.post("/api/claude/sessions/{sid}/stop")
    async def claude_stop(sid: int):
        get_or_404("claude_sessions", sid)
        await sessions_or_503().stop(sid)
        return {"ok": True}

    @app.post("/api/claude/sessions/{sid}/resume-at-reset")
    async def claude_resume_at_reset(sid: int, body: ResumeIn):
        row = get_or_404("claude_sessions", sid)
        if row["status"] != "paused_limit":
            raise HTTPException(400, "session is not paused by the usage limit")
        try:
            return tools.schedule_resume(con, row, body.message, body.at)
        except tools.ToolError as e:
            raise HTTPException(400, str(e))

    @app.delete("/api/claude/sessions/{sid}/resume-at-reset")
    async def claude_cancel_resume(sid: int):
        get_or_404("claude_sessions", sid)
        db.update(con, "claude_sessions", sid, resume_scheduled=0)
        return {"ok": True}

    # ---------- services ----------

    SUMMARIZE = ("Summarize this service's recent terminal output for its owner in 3-6 markdown bullets: is it "
                 "healthy, what is it doing, any errors or warnings (quote the key line), and a suggested next step.\n\n")

    def svc_fields(body) -> dict:
        """Reject malformed service bodies with 400 instead of letting a TypeError become a 500."""
        if not isinstance(body, dict):
            raise HTTPException(400, "body must be an object")
        for k in ("name", "tmux_session", "script_path", "cwd", "venv", "command", "notes"):
            if body.get(k) is not None and not isinstance(body[k], str):
                raise HTTPException(400, f"{k} must be a string")
        p = body.get("port")
        if p is not None and not (isinstance(p, int) and not isinstance(p, bool)
                                  or isinstance(p, str) and (p.isdigit() or p == "")):
            raise HTTPException(400, "port must be an integer")
        return body

    def body_str(body: dict, key: str, required=False) -> str | None:
        v = body.get(key)
        if v is not None and not isinstance(v, str):
            raise HTTPException(400, f"{key} must be a string")
        if required and not (v or "").strip():
            raise HTTPException(400, f"{key} is required")
        return v

    def svc_or_503() -> ServiceManager:
        if services is None:
            raise HTTPException(503, "The service manager is not available")
        return services

    async def svc_call(coro):
        try:
            return await coro
        except ServiceError as e:
            raise HTTPException(400, str(e))

    def svc_sync(fn, *a, **k):
        try:
            return fn(*a, **k)
        except ServiceError as e:
            raise HTTPException(400, str(e))

    @app.get("/api/services")
    async def services_status():
        return await svc_call(svc_or_503().status())

    @app.get("/api/gpu")
    async def gpu_status():
        return await svc_or_503().gpu()

    @app.post("/api/services/parse")
    async def services_parse(body: dict):
        try:
            return parse_script(body_str(body, "script_path") or "")
        except (OSError, ServiceError) as e:
            raise HTTPException(400, str(e))

    @app.get("/api/services/import-candidates")
    async def services_import_candidates():
        return svc_or_503().import_candidates()

    @app.post("/api/services")
    async def services_register(body: dict):
        return svc_sync(svc_or_503().register, svc_fields(body))

    @app.patch("/api/services/{name}")
    async def services_edit(name: str, body: dict):
        svc_fields(body)
        cur = svc_sync(svc_or_503().get, name)
        return svc_sync(svc_or_503().register, {**cur, **{k: v for k, v in body.items() if k != "name"}})

    @app.delete("/api/services/{name}")
    async def services_delete(name: str):
        svc_sync(svc_or_503().unregister, name)
        return {"ok": True}

    @app.get("/api/services/{name}/output")
    async def services_output(name: str, lines: int = 300):
        return {"output": await svc_call(svc_or_503().output(name, max(10, min(lines, 2000))))}

    @app.post("/api/services/{name}/send")
    async def services_send(name: str, body: dict):
        text = body_str(body, "text")
        return await svc_call(svc_or_503().send(name, text, bool(body.get("ctrl_c")), origin="ui"))

    @app.post("/api/services/{name}/start")
    async def services_start(name: str, body: dict = Body(default={})):
        return await svc_call(svc_or_503().start(name, override_gpu=bool(body.get("override_gpu"))))

    @app.post("/api/services/{name}/stop")
    async def services_stop(name: str):
        return await svc_call(svc_or_503().stop(name))

    @app.post("/api/services/{name}/restart")
    async def services_restart(name: str):
        return await svc_call(svc_or_503().restart(name))

    @app.post("/api/services/{name}/summarize")
    async def services_summarize(name: str):
        out = await svc_call(svc_or_503().output(name, 300))
        try:
            resp, _ = await llm.generate(SUMMARIZE + out[-12000:])
        except AllModelsExhausted:
            raise HTTPException(503, "Gemini is out of quota")
        except Exception as e:
            log.exception("summarize failed")
            raise HTTPException(502, f"Could not summarize: {type(e).__name__}")
        return {"summary": (resp.text or "").strip()}

    @app.post("/api/services/{name}/investigate")
    async def services_investigate(name: str, body: dict):
        return await run_tool("investigate_service",
                              {"name": name, "question": body_str(body, "question") or "Why is this failing?"})

    @app.get("/api/services/{name}/changes")
    async def services_changes(name: str):
        return svc_sync(svc_or_503().changes_for, name)

    @app.post("/api/services/{name}/change")
    async def services_change(name: str, body: dict):
        req = (body_str(body, "request") or "").strip()
        if not req:
            raise HTTPException(400, "describe the change")
        return await svc_call(svc_or_503().change_config(name, req))

    @app.get("/api/changes/{cid}")
    async def change_get(cid: int):
        return svc_sync(svc_or_503().change, cid)

    @app.post("/api/changes/{cid}/apply")
    async def change_apply(cid: int):
        return await svc_call(svc_or_503().apply(cid))

    @app.post("/api/changes/{cid}/undo")
    async def change_undo(cid: int):
        return await svc_call(svc_or_503().undo(cid))

    @app.get("/api/actions")
    async def actions_list():
        cutoff = db.to_utc(datetime.now(timezone.utc) - ACTION_TTL)
        return db.rows(con, "SELECT * FROM pending_actions WHERE status='pending' AND created_at >= ? ORDER BY id DESC",
                       cutoff)

    @app.post("/api/actions/{aid}/confirm")
    async def actions_confirm(aid: int):
        row = db.one(con, "SELECT kind FROM pending_actions WHERE id=?", aid)
        if row and row["kind"].startswith("repo_"):
            return await svc_call(repos_or_503().confirm_action(aid))
        return await svc_call(svc_or_503().confirm_action(aid))

    @app.post("/api/actions/{aid}/cancel")
    async def actions_cancel(aid: int):
        con.execute("UPDATE pending_actions SET status='cancelled', decided_at=? WHERE id=? AND status='pending'",
                    (db.now(), aid))
        return {"ok": True}

    # ---------- repos ----------

    def repos_or_503() -> RepoManager:
        if repos is None:
            raise HTTPException(503, "The repo workspace is not available")
        return repos

    @app.get("/api/repos")
    async def repos_list():
        return repos_or_503().list_repos()

    @app.post("/api/repos")
    async def repos_add(body: dict):
        url = body_str(body, "url", required=True)
        return await svc_call(repos_or_503().add_repo(url, body_str(body, "venv"), body_str(body, "name"),
                                                      body_str(body, "entry") or "main"))

    @app.get("/api/repos/{name}")
    async def repos_get(name: str):
        m = repos_or_503()
        v = svc_sync(m.view, name)
        if v["suite_status"] == "proposed":
            code, out = await m._exec(["git", "diff", "--stat", f"{v['default_branch']}...{SUITE_BRANCH}"], v["path"])
            v["suite_diff_stat"] = out[:5000] if code == 0 else ""
        return {**v, "runs": svc_sync(m.runs, name)}

    @app.post("/api/repos/{name}/env-check")
    async def repos_env_check(name: str):
        return await svc_call(repos_or_503().env_check(name))

    @app.post("/api/repos/{name}/suite")
    async def repos_suite(name: str, body: dict = Body(default={})):
        return await svc_call(repos_or_503().build_suite(name, body_str(body, "focus")))

    @app.post("/api/repos/{name}/suite/accept")
    async def repos_suite_accept(name: str):
        return await svc_call(repos_or_503().accept_suite(name))

    @app.post("/api/repos/{name}/test")
    async def repos_test(name: str, body: dict = Body(default={})):
        run = await svc_call(repos_or_503().run_tests(name, body_str(body, "ref")))
        return {k: v for k, v in run.items() if k != "results"} | \
            {"failing": [t for t, o in run["results"].items() if o in ("failed", "error")]}

    @app.post("/api/repos/{name}/tasks")
    async def repos_task_start(name: str, body: dict):
        return await svc_call(repos_or_503().start_task(name, body_str(body, "request") or ""))

    @app.get("/api/repos/{name}/tasks")
    async def repos_tasks(name: str):
        return svc_sync(repos_or_503().tasks, name)

    @app.get("/api/tasks/{tid}")
    async def task_get(tid: int):
        return await svc_call(repos_or_503().task_detail(tid))

    @app.post("/api/tasks/{tid}/fix")
    async def task_fix(tid: int):
        return await svc_call(repos_or_503().fix_task(tid))

    @app.post("/api/tasks/{tid}/merge")
    async def task_merge(tid: int):
        return await svc_call(repos_or_503().merge_task(tid))

    @app.post("/api/tasks/{tid}/discard")
    async def task_discard(tid: int):
        return await svc_call(repos_or_503().discard(tid))

    @app.post("/api/repos/{name}/undo-merge")
    async def repos_undo_merge(name: str):
        return await svc_call(repos_or_503().undo_merge(name))

    @app.post("/api/repos/{name}/pull")
    async def repos_pull(name: str):
        return await svc_call(repos_or_503().pull(name))

    @app.post("/api/repos/{name}/push")
    async def repos_push(name: str, body: dict = Body(default={})):
        return await svc_call(repos_or_503().push(name, body_str(body, "branch")))

    # ---------- web terminal (a full shell: login + extra unlock) ----------

    def unlock_state(token: str | None, touch: bool = False) -> bool:
        ts = unlocked.get(token) if token else None
        if ts is None:
            return False
        if time.time() - ts > terminals.idle_s:
            unlocked.pop(token, None)
            return False
        if touch:
            unlocked[token] = time.time()
        return True

    def term_or_503() -> TerminalManager:
        if terminals is None:
            raise HTTPException(503, "The web terminal is not available")
        return terminals

    def term_guard(request: Request) -> TerminalManager:
        m = term_or_503()  # the http middleware already required login
        if not unlock_state(login_token(request), touch=True):
            raise HTTPException(403, "terminal locked")
        return m

    async def term_call(coro):
        try:
            return await coro
        except TerminalError as e:
            raise HTTPException(400, str(e))

    @app.post("/api/terminal/unlock")
    async def terminal_unlock(body: LoginIn, request: Request):
        m = term_or_503()
        check_password(body.password, request)
        unlocked[login_token(request)] = time.time()
        return {"ok": True, "idle_min": round(m.idle_s / 60)}

    @app.get("/api/terminal/status")
    async def terminal_status(request: Request):
        return {"unlocked": terminals is not None and unlock_state(login_token(request))}

    @app.post("/api/terminal/lock")
    async def terminal_lock(request: Request):
        unlocked.pop(login_token(request), None)
        return {"ok": True}

    @app.get("/api/terminals")
    async def terminals_list(request: Request):
        return await term_call(term_guard(request).list())

    @app.post("/api/terminals")
    async def terminals_create(request: Request):
        return await term_call(term_guard(request).create())

    @app.delete("/api/terminals/{name}")
    async def terminals_close(name: str, request: Request):
        await term_call(term_guard(request).close(name))
        return {"ok": True}

    @app.websocket("/api/terminals/{name}/ws")
    async def terminal_ws(ws: WebSocket, name: str):
        # the http middleware does not run for websockets: every check is repeated here, before accept()
        token = login_token(ws)
        origin = urlparse(ws.headers.get("origin") or "").netloc.lower()  # blocks cross-site websocket hijacking
        if (terminals is None or not token or not origin or origin != (ws.headers.get("host") or "").lower()
                or not unlock_state(token, touch=True) or not valid_name(name) or not await terminals.exists(name)):
            await ws.close(4403)
            return
        ip = ws.client.host if ws.client else "?"
        try:
            att = await terminals.attach(name)
        except (TerminalError, OSError):
            await ws.close(4404)
            return
        await ws.accept()
        log.info("terminal %s connected from %s", name, ip)

        async def out():
            while (text := await att.read()) is not None:  # output is not activity: `tail -f` mustn't keep it unlocked
                await ws.send_text(text)

        async def inp():
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    return
                if not unlock_state(token, touch=True):
                    return await ws.close(4401)
                try:
                    ev = json.loads(msg.get("text") or "")
                    if ev["t"] == "in":
                        att.write(str(ev["d"]).encode())
                    elif ev["t"] == "resize":
                        att.resize(ev["cols"], ev["rows"])
                except (ValueError, KeyError, TypeError):
                    pass  # anything else is ignored

        async def watch():
            while unlock_state(token):
                await asyncio.sleep(min(2, max(terminals.idle_s / 2, 0.05)))  # 🔒 Lock / logout bite within 2 s
            await ws.close(4401)

        tasks = [asyncio.ensure_future(f()) for f in (out, inp, watch)]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in tasks:
                t.cancel()
            att.close()  # sync: kills + reaps the tmux client even if this handler is being cancelled
            await asyncio.gather(*tasks, return_exceptions=True)
            log.info("terminal %s disconnected from %s", name, ip)

    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(), logging.handlers.RotatingFileHandler(
                            config.ROOT / "assistant.log", maxBytes=5_000_000, backupCount=2)])
    missing = [k for k in ("GEMINI_API_KEY", "APP_PASSWORD", "SESSION_SECRET") if not getattr(config, k)]
    if missing:
        raise SystemExit(f"Missing in .env: {', '.join(missing)}")
    con = db.connect(config.DB_PATH)
    client = genai.Client(api_key=config.GEMINI_API_KEY, http_options=types.HttpOptions(
        timeout=120_000, retry_options=types.HttpRetryOptions(attempts=1)))
    llm = LLM(client, con, config.MODELS, config.EMBED_MODEL, config.SEARCH_MODELS, config.RPM_LIMIT)
    runner = SessionManager(con, timeout=config.CLAUDE_TIMEOUT, idle_s=config.CLAUDE_IDLE_MIN * 60,
                            max_running=config.CLAUDE_MAX_SESSIONS)
    services = ServiceManager(con, runner=runner, change_wait=config.CLAUDE_WAIT)
    repos = RepoManager(con, runner=runner, test_timeout=config.REPO_TEST_TIMEOUT)
    terminals = TerminalManager(idle_s=config.TERMINAL_IDLE_MIN * 60)
    uvicorn.run(create_app(con, llm, runner, services=services, repos=repos, terminals=terminals), host="0.0.0.0", port=config.PORT,
                timeout_graceful_shutdown=5)


if __name__ == "__main__":
    main()
