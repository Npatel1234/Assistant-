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
from services import ServiceError, parse_script

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
    services: object = None
    repos: object = None


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
    except (ToolError, ServiceError) as e:
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
    ps = [p for p in ps if norm(p["name"])]
    exact = [p for p in ps if norm(p["name"]) == n]
    return exact or [p for p in ps if n in norm(p["name"])]


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
    if fields.get("path"):
        p = Path(fields["path"]).expanduser()
        if not p.is_absolute():
            raise ToolError("path must be absolute")
        if not p.is_dir():
            raise ToolError(f"folder {fields['path']} does not exist on this machine")
        p = p.resolve()
        if p == Path("/") or p == Path.home().resolve():
            raise ToolError("refusing to use / or the home folder as a project")
        fields["path"] = str(p)


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
        raw = plan["steps"]
        name, goal, steps = plan["name"].strip(), plan["goal"], [str(s) for s in ([raw] if isinstance(raw, str) else raw)]
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
      "It runs as a live session the user can watch and type into in the Claude tab. "
      "Use for anything about the project's code. For pure questions, tell Claude not to modify files. "
      "follow_up=true continues the project's current Claude conversation.",
      {"project": S, "prompt": S, "follow_up": {"type": "boolean"}}, ["project", "prompt"])
async def ask_claude_code(ctx, project, prompt, follow_up=False):
    p = need_project(ctx.con, project)
    if not p["path"]:
        raise ToolError(f"Project '{p['name']}' has no folder path. Ask the user for it, "
                        "then call update_project with path.")
    res = await ctx.runner.run(p, prompt, follow_up, wait=config.CLAUDE_WAIT)
    return {"result": res["result"][:12000], "files_changed": res["files_changed"],
            "timed_out": res["timed_out"], "exit_code": res["exit_code"], "session": res.get("session"),
            "running": res.get("running", False)}


def schedule_resume(con, row: dict, message: str | None = None, at: str | None = None) -> dict:
    """Mark a usage-limit-paused session to resume at `at` (ISO local) or 2 min after its reset."""
    if at:
        when = parse_when(at)
    elif row["resume_scheduled"]:  # already scheduled: keep its time
        when = db.parse(row["resume_at"])
    else:
        when = db.parse(row["resume_at"]) + timedelta(minutes=2)
    msg = message or row["pending_message"]
    db.update(con, "claude_sessions", row["id"], resume_scheduled=1, resume_at=db.to_utc(when), pending_message=msg)
    return {"session": row["id"], "resume_local": fmt_local(db.to_utc(when)), "message": msg}


@tool("List Claude Code sessions with status (running / idle / paused by the usage limit with its reset time).")
async def list_claude_sessions(ctx):
    rows = db.rows(ctx.con, "SELECT s.id, p.name AS project, s.status, s.resume_at, s.resume_estimated, "
                            "s.resume_scheduled, s.pending_message, s.updated_at FROM claude_sessions s "
                            "JOIN projects p ON p.id = s.project_id ORDER BY s.updated_at DESC, s.id DESC LIMIT 20")
    for r in rows:
        r["resume_local"] = fmt_local(r["resume_at"]) if r["resume_at"] else None
        last = db.one(ctx.con, "SELECT output FROM claude_runs WHERE project_id=(SELECT project_id FROM "
                               "claude_sessions WHERE id=?) AND created_at >= (SELECT created_at FROM "
                               "claude_sessions WHERE id=?) ORDER BY id DESC LIMIT 1", r["id"], r["id"])
        r["last_result"] = (last["output"] or "")[:2000] if last else None
    return {"sessions": rows}


@tool("Schedule a Claude session that is paused by the usage limit to resume automatically. "
      "at = ISO local datetime (default: 2 minutes after the reported reset). message replaces the pending message.",
      {"project": S, "message": S, "at": S}, ["project"])
async def schedule_claude_resume(ctx, project, message=None, at=None):
    p = need_project(ctx.con, project)
    row = db.one(ctx.con, "SELECT * FROM claude_sessions WHERE project_id=? AND status='paused_limit' "
                          "ORDER BY id DESC LIMIT 1", p["id"])
    if not row:
        raise ToolError(f"No Claude session for {p['name']} is paused by the usage limit.")
    return schedule_resume(ctx.con, row, message, at)


@tool("Search the web (Google) for current info: libraries, papers, news, whether something already exists.",
      {"query": S}, ["query"])
async def web_search(ctx, query):
    resp, _ = await ctx.llm.search(query)
    gm = resp.candidates[0].grounding_metadata if resp.candidates else None
    chunks = (gm.grounding_chunks or []) if gm else []
    sources = [{"title": c.web.title, "url": c.web.uri} for c in chunks if c.web][:6]
    return {"answer": resp.text, "sources": sources}


def _svc(ctx):
    if ctx.services is None:
        raise ToolError("The service manager is not available")
    return ctx.services


@tool("List the user's services (tmux sessions): running?, port listening?, GPU use. Status only, no output; use service_output for a pane's output.")
async def list_services(ctx):
    st = await _svc(ctx).status()
    keys = ("name", "tmux_session", "registered", "managed", "alive", "port", "port_up", "should_run", "gpu_mb")
    return {"services": [{k: x.get(k) for k in keys} for x in st["services"]],
            "gpu": st.get("gpu"), "error": st.get("error")}


@tool("Recent output of a service's tmux pane.", {"name": S, "lines": I}, ["name"])
async def service_output(ctx, name, lines=200):
    out = await _svc(ctx).output(name, max(10, min(int(lines), 1000)))
    return {"name": name, "output": out[-8000:]}


@tool("Start a registered service (dependencies first; refuses if the GPU lacks memory — then the user can "
      "confirm an override).", {"name": S}, ["name"])
async def start_service(ctx, name):
    res = await _svc(ctx).start(name)
    if res.get("gpu_blocked"):
        res["override"] = _svc(ctx).create_action("start_override_gpu", name)
    return res


@tool("Stop a service. Returns an action the user must confirm in the app.", {"name": S}, ["name"])
async def stop_service(ctx, name):
    svc = _svc(ctx)._managed(name)
    return _svc(ctx).create_action("stop", svc["name"])


@tool("Restart a service. Returns an action the user must confirm in the app.", {"name": S}, ["name"])
async def restart_service(ctx, name):
    svc = _svc(ctx)._managed(name)
    return _svc(ctx).create_action("restart", svc["name"])


@tool("Type a command into a service's tmux pane. Returns an action the user must confirm in the app.",
      {"name": S, "text": S, "ctrl_c": {"type": "boolean"}}, ["name"])
async def send_to_service(ctx, name, text="", ctrl_c=False):
    svc = _svc(ctx)._managed(name)
    if text:
        live = (await _svc(ctx).sessions()).get(svc["tmux_session"])
        cmd = live["pane_command"].rsplit("/", 1)[-1] if live else ""
        if live and cmd not in {"bash", "zsh", "sh", "fish", "dash", "tmux"}:
            return {"error": f"{svc['tmux_session']} pane is running {cmd}; typed text would go to that program, "
                             "not a shell. Stop it first (ctrl_c), or use call_service / change_service_config."}
    return _svc(ctx).create_action("send", svc["name"], {"text": text, "ctrl_c": ctrl_c})


HTTP_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}


@tool("Call an HTTP endpoint of a registered service on this machine (uses its registered port). GET runs now; "
      "other methods return an action the user must confirm. Use this to test that a service works.",
      {"name": S, "path": {"type": "string", "description": "starts with /, e.g. /health"},
       "method": {"type": "string", "enum": sorted(HTTP_METHODS)}, "body": {"type": "object"}}, ["name", "path"])
async def call_service(ctx, name, path, method="GET", body=None):
    m = _svc(ctx)
    svc = m._managed(name)
    method = str(method).upper()
    if not svc["port"]:
        raise ToolError(f"service {svc['name']} has no port registered")
    if not path.startswith("/") or "://" in path or re.search(r"\s", path):
        raise ToolError("path must start with / and contain no spaces or scheme")
    if method not in HTTP_METHODS:
        raise ToolError(f"method must be one of {sorted(HTTP_METHODS)}")
    if body is not None and not isinstance(body, dict):
        raise ToolError("body must be a JSON object or null")
    if method == "GET":
        return await m.http(svc["name"], method, path, body)
    return m.create_action("http", svc["name"], {"method": method, "path": path, "body": body})


@tool("Register a service from its start script (run_*.sh). Returns the parsed fields and an action the user "
      "confirms in the app.", {"script_path": S, "notes": S}, ["script_path"])
async def register_service(ctx, script_path, notes=None):
    fields = parse_script(script_path)
    if notes:
        fields["notes"] = notes
    if not fields["ok"]:
        raise ToolError(f"Could not fully parse {script_path}: {fields}. Ask the user to register it in the Services tab.")
    return {"parsed": fields, **_svc(ctx).create_action("register", fields["name"], fields)}


@tool("Have Claude Code change a service's configuration (e.g. 'use Gemini as the LLM provider'). Backs up files "
      "first and returns the diff; the user applies (restarts) or undoes it in the app.",
      {"name": S, "request": S}, ["name", "request"])
async def change_service_config(ctx, name, request):
    c = await _svc(ctx).change_config(name, request)
    return {"change": c["id"], "request": c["request"], "status": c["status"], "flagged": bool(c["flagged"]),
            "flags": c["result"].get("flags", []), "diff": (c["diff"] or "")[:2500],
            "explanation": (c["explanation"] or "")[:1500]}


@tool("Undo a service config change (restores the original files; restarts the service if it was applied).",
      {"change_id": I}, ["change_id"])
async def undo_service_change(ctx, change_id):
    c = _svc(ctx).change(int(change_id))
    if c["status"] == "applied":
        raise ToolError("This change was applied; undoing it restarts the service — ask the user to tap Undo "
                        "in the Services tab.")
    return await _svc(ctx).undo(int(change_id))


@tool("Ask Claude Code to investigate a service using its recent tmux output and its code.",
      {"name": S, "question": S}, ["name", "question"])
async def investigate_service(ctx, name, question):
    svc = _svc(ctx).get(name)
    out = await _svc(ctx).output(name, 200)
    project = _svc(ctx).project_for(svc)
    prompt = (f"{question}\n\nRecent output of tmux session {svc['tmux_session']}:\n```\n{out[-6000:]}\n```\n"
              "Investigate using the code in this folder. Do not modify files unless the user asked for a fix.")
    res = await ctx.runner.run(project, prompt, False, wait=config.CLAUDE_WAIT)
    return {"result": res["result"][:12000], "session": res.get("session"), "running": res.get("running", False)}


def _repos(ctx):
    if ctx.repos is None:
        raise ToolError("The repo workspace is not available")
    return ctx.repos


def _task_brief(t: dict) -> dict:
    return {k: t.get(k) for k in ("id", "repo", "request", "branch", "status", "regressions", "new_tests",
                                  "fixed_tests", "new_failing", "before", "after", "error", "env_warning")} | \
        {"summary": (t.get("summary") or "")[:1500], "diff_stat": (t.get("diff_stat") or "")[:1500]}


@tool("Clone a Git/GitHub repository into the assistant's workspace so it can be tested and changed. "
      "venv = path of an existing Python venv (folder or python binary) that can run the app.",
      {"url": S, "venv": S, "name": S}, ["url"])
async def add_repo(ctx, url, venv=None, name=None):
    return await _repos(ctx).add_repo(url, venv, name)


@tool("Have Claude Code write the workspace repo's pytest suite (background job; the user reviews and accepts it).",
      {"repo": S, "focus": S}, ["repo"])
async def build_test_suite(ctx, repo, focus=None):
    return await _repos(ctx).build_suite(repo, focus)


@tool("Run a workspace repo's assistant test suite (on its main branch by default).", {"repo": S, "ref": S}, ["repo"])
async def run_repo_tests(ctx, repo, ref=None):
    run = await _repos(ctx).run_tests(repo, ref)
    failing = [t for t, o in run["results"].items() if o in ("failed", "error")]
    return {k: run[k] for k in ("ref", "passed", "failed", "errors", "skipped", "duration_s", "timed_out")} | \
        {"failing": failing[:30]}


@tool("Start a change request on a workspace repo: new branch, tests before, Claude makes the change, tests after, "
      "regressions and a behaviour summary. Runs in the background; check it with repo_task.",
      {"repo": S, "request": S}, ["repo", "request"])
async def start_repo_task(ctx, repo, request):
    return _task_brief(await _repos(ctx).start_task(repo, request))


@tool("Ask Claude to fix the regressions of a repo task (background).", {"task_id": I}, ["task_id"])
async def fix_task_regressions(ctx, task_id):
    return _task_brief(await _repos(ctx).fix_task(int(task_id)))


@tool("List repo tasks with status and regressions.", {"repo": S})
async def list_repo_tasks(ctx, repo=None):
    return {"tasks": [_task_brief(t) for t in _repos(ctx).tasks(repo)[:20]]}


@tool("Details of one repo task: status, tests before/after, regressions, behaviour summary.", {"task_id": I}, ["task_id"])
async def repo_task(ctx, task_id):
    return _task_brief(_repos(ctx).task(int(task_id)))


@tool("Merge a finished repo task into the local main branch and re-run the tests. Returns an action the user confirms.",
      {"task_id": I}, ["task_id"])
async def merge_task(ctx, task_id):
    m = _repos(ctx)  # create_action checks the task belongs to the repo and is ready
    return m.create_action("repo_merge", m.task(int(task_id))["repo"], {"task": int(task_id)})


@tool("Undo the last assistant merge in a workspace repo. Returns an action the user confirms.", {"repo": S}, ["repo"])
async def undo_merge(ctx, repo):
    return _repos(ctx).create_action("repo_undo_merge", repo)


@tool("Pull the latest commits from origin into a workspace repo's main branch and check for regressions.",
      {"repo": S}, ["repo"])
async def pull_repo(ctx, repo):
    return await _repos(ctx).pull(repo)


@tool("Push a branch of a workspace repo to origin (default: main). Returns an action the user confirms.",
      {"repo": S, "branch": S}, ["repo"])
async def push_repo(ctx, repo, branch=None):
    return _repos(ctx).create_action("repo_push", repo, {"branch": branch})


@tool("Discard a repo task's branch (not for merged tasks).", {"task_id": I}, ["task_id"])
async def discard_task(ctx, task_id):
    return _task_brief(await _repos(ctx).discard(int(task_id)))


REPO_TOOLS = {"add_repo", "build_test_suite", "run_repo_tests", "start_repo_task", "fix_task_regressions",
              "list_repo_tasks", "repo_task", "merge_task", "undo_merge", "pull_repo", "push_repo", "discard_task"}
