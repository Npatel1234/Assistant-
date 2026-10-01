# Repo Workspace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clone GitHub repos into `workspace/`, have Claude build a pytest suite for each, run change requests as branch-per-task work with before/after test comparison and a behaviour summary, and merge/pull with regression checks; push only after the user confirms.

**Architecture:** New `workspace.py` (`RepoManager`) drives git and pytest through asyncio subprocesses (own process group, timeout kill), runs Claude work through the existing `SessionManager`, and shares the `pending_actions` confirmation system (claim/finish helpers extracted from `services.py`). Long operations (suite build, tasks, fixes) run as background tasks; the UI polls.

**Tech Stack:** Python 3.11 asyncio, git CLI, pytest + JUnit XML, SQLite, FastAPI, vanilla JS.

**Spec:** `docs/superpowers/specs/2026-09-30-repo-workspace-design.md`

## Global Constraints

- No new dependencies. Tests: `uv run pytest -q test_core.py -W error`, pristine. The new tests use real `git` on throwaway repos under `tmp_path` and never touch `workspace/RAG_for_192_server`, the user's real projects, or GitHub.
- All repo paths resolve inside the RepoManager root (production: `config.ROOT / "workspace"`); repo names match `[\w.-]+` and must not start with `.`.
- Never `git push --force`/`--force-with-lease`; `git reset --hard` only in undo-merge, guarded by `HEAD == merge_sha`. Branch names are generated (`task/<id>-<slug>`, `assistant/tests`).
- Operations that change a clone require a clean worktree (`git status --porcelain` empty, ignoring `__pycache__/` and `.pytest_cache/`); never stash.
- The user's venv is never modified except by a confirmed `repo_install` (`<venv python> -m pip install <pkg>`, pkg matches `^[A-Za-z0-9._-]+(==[\w.]+)?$`). pytest comes from an overlay dir (`<root>/.testtools/py<ver>/`) only when the venv lacks it.
- Test runs: `<venv python> -m pytest tests_assistant -q -p no:cacheprovider --junitxml=<tmp>/report.xml -o junit_family=xunit2`, env `ASSISTANT_TEST=1`, `PYTHONDONTWRITEBYTECODE=1`, `PYTHONPATH=<overlay>:<repo>`, cwd = repo, own process group, killed on timeout. Temp dirs are created under `<root>/.runs/` (never rely on `tempfile.gettempdir()` — tests patch it).
- Gemini-initiated merge / undo-merge / push / install are pending actions (`repo_merge`, `repo_undo_merge`, `repo_push`, `repo_install`) executed only via `POST /api/actions/{id}/confirm`.
- Tools that run Claude (`add_repo` excluded — it only clones) are guarded by the injection guard: `build_test_suite`, `start_repo_task`, `fix_task_regressions`, `pull_repo`, plus `add_repo` (it fetches untrusted code).
- Timestamps UTC via `db.now()`. Commit messages end with a blank line then `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Modify files with Write/Edit only — no bash heredocs. Never read/print/commit `.env`.

## Review Focus

1. **Suite merged locally → local main diverges from origin** → pull must merge upstream (not ff-only) and abort cleanly on conflict; test `test_pull_detects_upstream_regression` (Task 1).
2. **Undo-merge after main moved** → refused, nothing reset; test `test_undo_refused_when_main_moved` (Task 1).
3. **A hanging test** → the whole process group is killed at the timeout and the task fails with "timed out"; test `test_task_test_timeout_fails_task` (Task 1).
4. **Two operations on one repo at once** (e.g. a task while the suite builds) → refused, repo state intact; test `test_one_operation_per_repo` (Task 1).
5. **Gemini merging/pushing without a tap** → tools only create pending actions; test `test_repo_tools_create_actions` (Task 2).

---

## File Structure

| File | Change |
|---|---|
| `workspace.py` | **new**: `RepoError`, `parse_junit`, `compare_runs`, `RepoManager` |
| `services.py` | extract `claim_action` / `finish_action`; `ServiceManager.confirm_action` uses them |
| `db.py` | tables `repos`, `repo_tasks`, `test_runs` |
| `tools.py` | `Ctx.repos`; 12 repo tools |
| `agent.py` | guard + taint sets; greeting lists finished repo tasks and upstream regressions |
| `main.py` | wire `RepoManager`; repos/tasks API; action confirm dispatch |
| `static/index.html` | 🐙 Repos tab + task cards |
| `README.md` | Repos section |
| `test_core.py` | tests |

---

### Task 1: RepoManager core

**Files:** Create `workspace.py`; modify `services.py`, `db.py`; test `test_core.py` (append).

**Interfaces (produced):**
- `services.claim_action(con, aid) -> dict` (row with `payload` parsed; raises `ServiceError` if not pending / expired → marks expired), `services.finish_action(con, aid, result=None, error=None)`.
- `workspace.RepoError(ServiceError)`, `workspace.parse_junit(path) -> {nodeid: outcome}`, `workspace.compare_runs(before, after, known) -> {"regressions","new_tests","fixed_tests","new_failing"}`.
- `workspace.RepoManager(con, runner=None, root=None, test_timeout=300, claude_attempts=3)`:
  `get(name)`, `view(name)`, `list_repos()`, `task(tid)`, `async task_detail(tid)` (+ `diff`), `tasks(name=None)`, `runs(name, limit=20)`,
  `async add_repo(url, venv=None, name=None, entry="main")`, `async env_check(name)`,
  `async build_suite(name, focus=None)` (background), `async accept_suite(name)`,
  `async run_tests(name, ref=None)`, `async start_task(name, request)` (background), `async fix_task(tid)` (background),
  `async merge_task(tid)`, `async undo_merge(name)`, `async pull(name)`, `async push(name, branch=None)`, `async discard(tid)`,
  `async install_package(name, package)`, `create_action(kind, repo, payload=None, created_by="gemini")`, `async confirm_action(aid)`, `async wait_idle()`.

- [ ] **Step 1: Refactor pending-action helpers in `services.py`**

Add at module level (below `_udiff`):
```python
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


def finish_action(con, aid: int, result=None, error: BaseException | None = None) -> None:
    if error is not None:
        db.update(con, "pending_actions", aid, status="failed", result=str(error) or type(error).__name__,
                  decided_at=db.now())
    else:
        db.update(con, "pending_actions", aid, status="done", result=json.dumps(result, default=str)[:4000],
                  decided_at=db.now())
```
(`ServiceError` is defined above `ServiceManager`; place these two functions after the `ServiceError` class.) Replace the body of `ServiceManager.confirm_action` with:
```python
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
            else:
                raise ServiceError(f"{row['kind']} is not a service action")
        except Exception as e:
            finish_action(self.con, aid, error=e)
            raise
        finish_action(self.con, aid, result=res)
        return res
```
Run `uv run pytest -q test_core.py -W error` → still green (behaviour unchanged).

- [ ] **Step 2: Tables in `db.py`** — append inside `SCHEMA` (after `pending_actions`):
```sql
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
```

- [ ] **Step 3: Append failing tests to `test_core.py`**

```python
from workspace import RepoError, RepoManager, compare_runs, parse_junit

TINY_APP = '''from fastapi import FastAPI

app = FastAPI()


@app.get("/add")
def add(a: int, b: int):
    return {"sum": a + b}


@app.get("/hello")
def hello(name: str = "x"):
    return {"msg": f"hello {name}"}
'''

FAKE_REPO_CLAUDE = r'''#!/usr/bin/env python3
import json, os, sys
sid = f"repo-{os.getpid()}"
TESTS = """from fastapi.testclient import TestClient
import main

client = TestClient(main.app)


def test_add():
    assert client.get("/add", params={"a": 2, "b": 3}).json() == {"sum": 5}


def test_hello():
    assert client.get("/hello", params={"name": "bob"}).json() == {"msg": "hello bob"}
"""
MUL = """

@app.get("/mul")
def mul(a: int, b: int):
    return {"product": a * b}
"""
MUL_TEST = """from fastapi.testclient import TestClient
import main


def test_mul():
    assert TestClient(main.app).get("/mul", params={"a": 2, "b": 3}).json() == {"product": 6}
"""
def out(o):
    print(json.dumps(o), flush=True)
def write(path, text):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    open(path, "w").write(text)
out({"type": "system", "subtype": "init", "session_id": sid})
for line in sys.stdin:
    msg = json.loads(line)["message"]["content"]
    reply = "ok"
    if "Write a pytest suite" in msg:
        write("tests_assistant/test_api.py", TESTS)
        if "with-broken" in msg:
            write("tests_assistant/test_known.py", "def test_known():\n    assert False\n")
        reply = "wrote the suite"
    elif "TASK:" in msg:
        src = open("main.py").read()
        if "break-add" in msg:
            src = src.replace("a + b}", "a + b + 1}")
        if "alt-add" in msg:
            src = src.replace("a + b}", "a + b + 2}")
        if "add-mul" in msg:
            src += MUL
            write("tests_assistant/test_mul.py", MUL_TEST)
        if "slow-test" in msg:
            write("tests_assistant/test_slow.py", "import time\n\n\ndef test_slow():\n    time.sleep(60)\n")
        open("main.py", "w").write(src)
        reply = "NEEDS_PACKAGE: requestsx==1.0" if "need-pkg" in msg else "changed main.py"
    elif "Fix the failing tests caused by your change" in msg:
        src = open("main.py").read()
        open("main.py", "w").write(src.replace("a + b + 1}", "a + b}").replace("a + b + 2}", "a + b}"))
        reply = "fixed"
    elif "Summarize" in msg:
        reply = "- the add endpoint behaviour changed"
    out({"type": "assistant", "message": {"content": [{"type": "text", "text": reply}]}, "session_id": sid})
    out({"type": "result", "is_error": False, "result": reply, "session_id": sid})
'''

_GIT_ID = ["-c", "user.name=Test", "-c", "user.email=test@example.com"]


def _git(cwd, *args):
    return subprocess.run(["git", *_GIT_ID, *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo_env(con, tmp_path):
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "main.py").write_text(TINY_APP)
    _git(seed, "init", "-q", "-b", "main")
    _git(seed, "add", "-A")
    _git(seed, "commit", "-q", "-m", "init")
    origin = tmp_path / "origin.git"
    _git(tmp_path, "clone", "-q", "--bare", str(seed), str(origin))
    _git(seed, "remote", "add", "origin", str(origin))
    fake = tmp_path / "claude-repo"
    fake.write_text(FAKE_REPO_CLAUDE)
    fake.chmod(0o755)
    ws = tmp_path / "ws"

    def make(**kw):
        runner = SessionManager(con, claude_bin=str(fake), timeout=60)
        return RepoManager(con, runner=runner, root=ws, **{"test_timeout": 60, "claude_attempts": 2, **kw}), runner

    return origin, seed, make


async def _ready_repo(m, origin, focus=None):
    await m.add_repo(str(origin), name="tiny")
    await m.build_suite("tiny", focus)
    await m.wait_idle()
    await m.accept_suite("tiny")


async def _task(m, request):
    t = await m.start_task("tiny", request)
    await m.wait_idle()
    return m.task(t["id"])


def test_compare_runs_and_junit(tmp_path):
    before = {"a::t1": "passed", "a::t2": "passed", "a::t3": "failed", "a::k": "passed"}
    after = {"a::t1": "failed", "a::t2": "passed", "a::t3": "passed", "a::t4": "error", "a::k": "failed"}
    c = compare_runs(before, after, ["a::k"])
    assert c == {"regressions": ["a::t1"], "new_tests": ["a::t4"], "fixed_tests": ["a::t3"], "new_failing": ["a::t4"]}
    xml = tmp_path / "r.xml"
    xml.write_text('<testsuites><testsuite><testcase classname="tests.test_x" name="test_a"/>'
                   '<testcase classname="tests.test_x" name="test_b"><failure message="x"/></testcase>'
                   '<testcase classname="tests.test_x" name="test_c"><skipped/></testcase></testsuite></testsuites>')
    assert parse_junit(xml) == {"tests.test_x::test_a": "passed", "tests.test_x::test_b": "failed",
                                "tests.test_x::test_c": "skipped"}


def test_add_repo_clone_reuse_and_errors(con, repo_env):
    origin, _, make = repo_env

    async def body():
        m, runner = make()
        v = await m.add_repo(str(origin), name="tiny")
        again = await m.add_repo(str(origin), name="tiny")
        with pytest.raises(RepoError):
            await m.add_repo("ftp://nope/x.git")
        with pytest.raises(RepoError):
            await m.add_repo(str(origin), name="../escape")
        with pytest.raises(RepoError):
            await m.add_repo(str(origin), name=".testtools")
        with pytest.raises(RepoError):
            await m.add_repo("/definitely/missing/repo.git", name="missing")
        await runner.shutdown()
        return v, again

    v, again = asyncio.run(body())
    assert v["default_branch"] == "main" and v["env_status"] == "ok" and again["path"] == v["path"]
    assert Path(v["path"]).is_dir() and Path(v["path"]).parent.name == "ws"
    assert db.one(con, "SELECT path FROM projects WHERE path=?", v["path"])


def test_build_suite_and_accept(con, repo_env):
    origin, _, make = repo_env

    async def body():
        m, runner = make()
        await m.add_repo(str(origin), name="tiny")
        await m.build_suite("tiny")
        await m.wait_idle()
        proposed = m.view("tiny")
        acc = await m.accept_suite("tiny")
        v = m.view("tiny")
        await runner.shutdown()
        return proposed, acc, v

    proposed, acc, v = asyncio.run(body())
    assert proposed["suite_status"] == "proposed" and proposed["known_failures"] == []
    assert acc["baseline"]["passed"] == 2 and v["suite_status"] == "accepted"
    assert v["last_main_run"]["passed"] == 2 and (Path(v["path"]) / "tests_assistant" / "test_api.py").exists()
    assert _git(v["path"], "rev-parse", "--abbrev-ref", "HEAD") == "main"


def test_task_regression_then_fix(con, repo_env):
    origin, _, make = repo_env

    async def body():
        m, runner = make()
        await _ready_repo(m, origin)
        t = await _task(m, "break-add please")
        await m.fix_task(t["id"])
        await m.wait_idle()
        fixed = m.task(t["id"])
        await runner.shutdown()
        return t, fixed

    t, fixed = asyncio.run(body())
    assert t["status"] == "ready" and t["regressions"] == ["tests_assistant.test_api::test_add"]
    assert t["summary"].startswith("- the add endpoint") and t["branch"].startswith(f"task/{t['id']}-break-add")
    assert fixed["status"] == "ready" and fixed["regressions"] == []


def test_task_new_test_no_regression(con, repo_env):
    origin, _, make = repo_env

    async def body():
        m, runner = make()
        await _ready_repo(m, origin)
        t = await _task(m, "add-mul endpoint")
        await runner.shutdown()
        return t

    t = asyncio.run(body())
    assert t["regressions"] == [] and t["new_tests"] == ["tests_assistant.test_mul::test_mul"]
    assert t["after"]["passed"] == 3 and "main.py" in t["diff_stat"]


def test_merge_check_and_undo(con, repo_env):
    origin, _, make = repo_env

    async def body():
        m, runner = make()
        await _ready_repo(m, origin)
        t = await _task(m, "break-add please")
        before_head = _git(m.view("tiny")["path"], "rev-parse", "main")
        merged = await m.merge_task(t["id"])
        undo = await m.undo_merge("tiny")
        after_head = _git(m.view("tiny")["path"], "rev-parse", "main")
        state = m.task(t["id"])
        await runner.shutdown()
        return merged, before_head, after_head, state

    merged, before_head, after_head, state = asyncio.run(body())
    assert merged["check"]["vs_main"]["regressions"] == ["tests_assistant.test_api::test_add"]
    assert after_head == before_head and state["status"] == "ready"


def test_undo_refused_when_main_moved(con, repo_env):
    origin, _, make = repo_env

    async def body():
        m, runner = make()
        await _ready_repo(m, origin)
        t = await _task(m, "add-mul endpoint")
        await m.merge_task(t["id"])
        path = m.view("tiny")["path"]
        Path(path, "NOTES.txt").write_text("user commit")
        _git(path, "add", "-A")
        _git(path, "commit", "-q", "-m", "user work")
        head = _git(path, "rev-parse", "main")
        with pytest.raises(RepoError, match="moved"):
            await m.undo_merge("tiny")
        still = _git(path, "rev-parse", "main")
        await runner.shutdown()
        return head, still

    head, still = asyncio.run(body())
    assert head == still


def test_merge_conflict_aborted(con, repo_env):
    origin, _, make = repo_env

    async def body():
        m, runner = make()
        await _ready_repo(m, origin)
        t1 = await _task(m, "break-add please")
        t2 = await _task(m, "alt-add please")
        await m.merge_task(t1["id"])
        with pytest.raises(RepoError, match="conflict"):
            await m.merge_task(t2["id"])
        path = m.view("tiny")["path"]
        status = _git(path, "status", "--porcelain")
        state = m.task(t2["id"])
        await runner.shutdown()
        return status, state

    status, state = asyncio.run(body())
    assert status == "" and state["status"] == "ready" and "conflict" in state["error"]


def test_pull_detects_upstream_regression(con, repo_env):
    origin, seed, make = repo_env

    async def body():
        m, runner = make()
        await _ready_repo(m, origin)
        nothing = await m.pull("tiny")
        src = (seed / "main.py").read_text().replace("a + b}", "a - b}")
        (seed / "main.py").write_text(src)
        _git(seed, "commit", "-q", "-am", "upstream breaks add")
        _git(seed, "push", "-q", "origin", "main")
        res = await m.pull("tiny")
        await runner.shutdown()
        return nothing, res, m.view("tiny")

    nothing, res, v = asyncio.run(body())
    assert nothing["changed"] is False
    assert res["changed"] and res["regressions"] == ["tests_assistant.test_api::test_add"]
    assert any("upstream breaks add" in c for c in res["commits"]) and v["last_pull"]["regressions"]


def test_push_only_after_confirm(con, repo_env):
    origin, _, make = repo_env

    async def body():
        m, runner = make()
        await _ready_repo(m, origin)
        t = await _task(m, "add-mul endpoint")
        a = m.create_action("repo_push", "tiny", {"branch": t["branch"]})
        before = subprocess.run(["git", "branch", "--list", t["branch"]], cwd=origin, capture_output=True, text=True).stdout
        res = await m.confirm_action(a["action_id"])
        after = subprocess.run(["git", "branch", "--list", t["branch"]], cwd=origin, capture_output=True, text=True).stdout
        with pytest.raises(RepoError):
            await m.push("tiny", "--force")
        await runner.shutdown()
        return a, before, res, after

    a, before, res, after = asyncio.run(body())
    assert a["needs_confirmation"] and before.strip() == "" and res["pushed"] and after.strip()


def test_discard_and_unclean_refused(con, repo_env):
    origin, _, make = repo_env

    async def body():
        m, runner = make()
        await _ready_repo(m, origin)
        t = await _task(m, "add-mul endpoint")
        await m.discard(t["id"])
        branches = _git(m.view("tiny")["path"], "branch", "--list")
        Path(m.view("tiny")["path"], "main.py").write_text("# dirty\n")
        with pytest.raises(RepoError, match="uncommitted"):
            await m.start_task("tiny", "add-mul again")
        await runner.shutdown()
        return t, m.task(t["id"]), branches

    t, state, branches = asyncio.run(body())
    assert state["status"] == "discarded" and t["branch"] not in branches


def test_task_test_timeout_fails_task(con, repo_env):
    origin, _, make = repo_env

    async def body():
        m, runner = make(test_timeout=3)
        await _ready_repo(m, origin)
        start = time.monotonic()
        t = await _task(m, "slow-test please")
        await runner.shutdown()
        return t, time.monotonic() - start

    t, elapsed = asyncio.run(body())
    assert t["status"] == "failed" and "timed out" in t["error"] and elapsed < 40


def test_needs_package_creates_install_action(con, repo_env):
    origin, _, make = repo_env

    async def body():
        m, runner = make()
        await _ready_repo(m, origin)
        t = await _task(m, "need-pkg please")
        await runner.shutdown()
        return t

    t = asyncio.run(body())
    assert t["status"] == "needs_package" and "requestsx==1.0" in t["error"]
    row = db.one(con, "SELECT kind, payload FROM pending_actions ORDER BY id DESC LIMIT 1")
    assert row["kind"] == "repo_install" and json.loads(row["payload"])["package"] == "requestsx==1.0"


def test_known_failures_excluded(con, repo_env):
    origin, _, make = repo_env

    async def body():
        m, runner = make()
        await _ready_repo(m, origin, focus="with-broken")
        v = m.view("tiny")
        t = await _task(m, "add-mul endpoint")
        await runner.shutdown()
        return v, t

    v, t = asyncio.run(body())
    assert v["known_failures"] == ["tests_assistant.test_known::test_known"] and t["regressions"] == []


def test_one_operation_per_repo(con, repo_env):
    origin, _, make = repo_env

    async def body():
        m, runner = make()
        await m.add_repo(str(origin), name="tiny")
        await m.build_suite("tiny")
        with pytest.raises(RepoError, match="busy|running"):
            await m.build_suite("tiny")
        await m.wait_idle()
        await m.accept_suite("tiny")
        await m.start_task("tiny", "add-mul endpoint")
        with pytest.raises(RepoError, match="running"):
            await m.start_task("tiny", "break-add please")
        await m.wait_idle()
        await runner.shutdown()

    asyncio.run(body())
```

- [ ] **Step 4: Run to verify failure** — `uv run pytest -q test_core.py -k "compare_runs or add_repo or build_suite or task_ or merge or undo_refused or pull_detects or push_only or discard or needs_package or known_failures or one_operation"` → FAIL (`No module named 'workspace'`).

- [ ] **Step 5: Create `workspace.py`**

```python
"""Repo workspace: clone GitHub repos into workspace/, Claude-built pytest suites, one branch per change
request with before/after test comparison, local merge/pull with regression checks, confirmed push.
The user's venv is never modified except by a confirmed package install."""
import asyncio
import contextlib
import json
import logging
import os
import re
import shutil
import signal
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import config
import db
from services import ServiceError, claim_action, finish_action

log = logging.getLogger("workspace")
ACTIVE = ("queued", "testing_before", "claude", "testing_after")
SUITE_DIR = "tests_assistant"
SUITE_BRANCH = "assistant/tests"
PKG_RE = re.compile(r"^[A-Za-z0-9._-]+(==[\w.]+)?$")
NEEDS_RE = re.compile(r"NEEDS_PACKAGE:\s*([A-Za-z0-9._-]+(?:==[\w.]+)?)")
URL_RE = re.compile(r"^(https://|git@|file://|/)")
REPO_ACTIONS = {"repo_merge", "repo_undo_merge", "repo_push", "repo_install"}

SUITE_PROMPT = """Write a pytest suite for this repository in the folder tests_assistant/ (create it; put only test
files and an optional conftest.py there).
Goal: exercise every HTTP endpoint of the application (module `{entry}`, e.g. with fastapi.testclient.TestClient)
and its key helper functions.
Rules:
- Fake every LLM, embedding model, vector store, database and network call (monkeypatch / unittest.mock):
  no GPU, no model downloads, no outbound network. ASSISTANT_TEST=1 is set while the tests run.
- Deterministic and fast: the whole suite must finish in under 60 seconds.
- Do not modify files outside tests_assistant/. Do not pip install anything.
- Run it with: {cmd}
  and iterate until every test passes on the current code.
{focus}
When done, reply with the number of tests and anything you could not cover."""

FIX_SUITE_PROMPT = """Make the failing tests in tests_assistant/ pass on the current application code without changing
the application (fix the tests or their fakes). Failing tests:
{failures}
Run with: {cmd}"""

TASK_PROMPT = """TASK: {request}

You are on branch {branch} of this repository. Make the change, then add or update tests in tests_assistant/ that
cover the new or changed behaviour.
Rules:
- Keep existing tests passing unless the requested change intentionally alters that behaviour (then update those
  tests and say so).
- Do not pip install anything. If the change needs a package that is not installed, reply with a line
  "NEEDS_PACKAGE: <name>" and stop.
- Do not push, do not switch branches, do not touch other repositories.
- Run the suite with: {cmd}
- Commit your work on this branch when done."""

FIX_TASK_PROMPT = """Fix the failing tests caused by your change (they passed before it):
{failures}
Keep the requested behaviour. Run with: {cmd}. Commit when done."""

SUMMARY_PROMPT = """Summarize in 3-6 markdown bullets how the application's behaviour changed on this branch compared
to {base} (what users or API clients will notice: endpoints, responses, config, performance, risks). Do not change
any files. `git diff {base}...HEAD` shows the change."""

UPSTREAM_PROMPT = """Summarize in 3-6 markdown bullets how the application's behaviour changed between commits {old}
and {new} (run `git log -p {old}..{new}`). Point out anything that could break clients. Do not change any files."""


class RepoError(ServiceError):
    pass


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "task"


def parse_junit(path: Path) -> dict[str, str]:
    results: dict[str, str] = {}
    for tc in ET.parse(path).getroot().iter("testcase"):
        outcome = "passed"
        for child in tc:
            if child.tag == "failure":
                outcome = "failed"
            elif child.tag == "error":
                outcome = "error"
            elif child.tag == "skipped":
                outcome = "skipped"
        results[f"{tc.get('classname', '')}::{tc.get('name', '')}"] = outcome
    return results


def compare_runs(before: dict, after: dict, known: list[str]) -> dict:
    bad = {"failed", "error"}
    new = sorted(t for t in after if t not in before)
    return {"regressions": sorted(t for t, o in after.items()
                                  if o in bad and before.get(t) == "passed" and t not in known),
            "new_tests": new,
            "fixed_tests": sorted(t for t, o in after.items() if o == "passed" and before.get(t) in bad),
            "new_failing": sorted(t for t in new if after[t] in bad and t not in known)}


def _brief(run: dict | None) -> dict | None:
    if not run:
        return None
    return {k: run.get(k) for k in ("id", "ref", "sha", "kind", "passed", "failed", "errors", "skipped",
                                    "duration_s", "created_at", "timed_out")}


class RepoManager:
    def __init__(self, con, runner=None, root=None, test_timeout: float = 300, claude_attempts: int = 3):
        self.con, self.runner = con, runner
        self.root = Path(root or config.ROOT / "workspace").resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.test_timeout, self.claude_attempts = test_timeout, claude_attempts
        self._locks: dict[int, asyncio.Lock] = {}
        self._bg: set[asyncio.Task] = set()
        con.execute(f"UPDATE repo_tasks SET status='failed', error='interrupted by restart', updated_at=? "
                    f"WHERE status IN {ACTIVE}", (db.now(),))
        con.execute("UPDATE repos SET suite_status='failed', suite_error='interrupted by restart', updated_at=? "
                    "WHERE suite_status='building'", (db.now(),))

    # ---------- plumbing ----------
    def _lock(self, repo_id: int) -> asyncio.Lock:
        return self._locks.setdefault(repo_id, asyncio.Lock())

    def _spawn(self, coro) -> asyncio.Task:
        t = asyncio.create_task(coro)
        self._bg.add(t)
        t.add_done_callback(self._bg.discard)
        return t

    async def wait_idle(self) -> None:
        while self._bg:
            await asyncio.gather(*list(self._bg), return_exceptions=True)

    def _need_runner(self) -> None:
        if self.runner is None:
            raise RepoError("Claude sessions are not available")

    async def _exec(self, args: list[str], cwd, timeout: float = 120, env: dict | None = None) -> tuple[int, str]:
        full_env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", **(env or {})}
        try:
            proc = await asyncio.create_subprocess_exec(
                *args, cwd=str(cwd), env=full_env, start_new_session=True, stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        except FileNotFoundError:
            raise RepoError(f"{args[0]} is not installed")
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout)
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)  # the test runner and everything it spawned
            await proc.wait()
            return 124, f"timed out after {timeout:g}s"
        return proc.returncode, out.decode(errors="replace")

    async def git(self, path, *args, timeout: float = 120) -> str:
        code, out = await self._exec(["git", *args], path, timeout)
        if code != 0:
            raise RepoError(f"git {args[0]} failed: {out.strip()[-1500:]}")
        return out.strip()

    async def _ident(self, path) -> list[str]:
        code, out = await self._exec(["git", "config", "user.email"], path)
        return [] if code == 0 and out.strip() else ["-c", "user.name=Assistant", "-c", "user.email=assistant@localhost"]

    async def _clean(self, path) -> None:
        out = await self.git(path, "status", "--porcelain")
        dirty = [x for x in out.splitlines() if not re.search(r"(__pycache__|\.pytest_cache)/?$", x)]
        if dirty:
            raise RepoError("the clone has uncommitted changes:\n" + "\n".join(dirty[:20]))

    async def _commit_all(self, path, message: str) -> bool:
        if not await self.git(path, "status", "--porcelain"):
            return False
        await self.git(path, "add", "-A")
        await self.git(path, *(await self._ident(path)), "commit", "-q", "-m", message)
        return True

    async def _back_to_default(self, r: dict) -> None:
        path = Path(r["path"])
        if not await self.git(path, "status", "--porcelain"):
            await self.git(path, "checkout", "-q", r["default_branch"])

    @staticmethod
    def _python(venv) -> str:
        if not venv:
            return sys.executable
        p = Path(venv).expanduser()
        if p.is_dir():
            p = p / "bin" / "python"
        if not p.exists():
            raise RepoError(f"no python found at {p}")
        return str(p)

    async def _overlay(self, python: str) -> str | None:
        code, _ = await self._exec([python, "-c", "import pytest"], self.root, timeout=60)
        if code == 0:
            return None
        code, ver = await self._exec([python, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"], self.root, 60)
        target = self.root / ".testtools" / f"py{ver.strip()}"
        if not (target / "pytest").exists():
            uv = shutil.which("uv")
            if not uv:
                raise RepoError("pytest is missing in that venv and uv is not available to add it")
            code, out = await self._exec([uv, "pip", "install", "--quiet", "--python", python, "--target",
                                          str(target), "pytest"], self.root, timeout=600)
            if code != 0:
                raise RepoError(f"could not prepare test tools: {out.strip()[-1500:]}")
        return str(target)

    @staticmethod
    def _test_env(r: dict, overlay: str | None) -> dict:
        return {"ASSISTANT_TEST": "1", "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": os.pathsep.join(x for x in (overlay, r["path"]) if x)}

    def _cmd(self, r: dict, overlay: str | None) -> str:
        env = " ".join(f"{k}={v}" for k, v in self._test_env(r, overlay).items())
        return f"{env} {r['venv_python']} -m pytest {SUITE_DIR} -q -p no:cacheprovider"

    # ---------- rows & views ----------
    def get(self, name: str) -> dict:
        r = db.one(self.con, "SELECT * FROM repos WHERE name=?", name)
        if not r:
            known = ", ".join(x["name"] for x in db.rows(self.con, "SELECT name FROM repos ORDER BY name"))
            raise RepoError(f"No workspace repo named '{name}'. Known: {known or 'none'}.")
        return r

    def _row(self, rid: int) -> dict:
        return db.one(self.con, "SELECT * FROM repos WHERE id=?", rid)

    def _task_row(self, tid: int) -> dict:
        t = db.one(self.con, "SELECT * FROM repo_tasks WHERE id=?", tid)
        if not t:
            raise RepoError(f"task {tid} not found")
        return t

    def _load_run(self, run_id) -> dict | None:
        r = db.one(self.con, "SELECT * FROM test_runs WHERE id=?", run_id) if run_id else None
        if r:
            r["results"] = json.loads(r["results"] or "{}")
        return r

    def _last_main_run(self, r: dict) -> dict | None:
        row = db.one(self.con, "SELECT id FROM test_runs WHERE repo_id=? AND ref=? ORDER BY id DESC LIMIT 1",
                     r["id"], r["default_branch"])
        return self._load_run(row["id"]) if row else None

    def _project_row(self, r: dict) -> dict:
        p = db.one(self.con, "SELECT * FROM projects WHERE id=?", r["project_id"])
        if not p:
            raise RepoError("the repo's project row is missing; add the repo again")
        return p

    def view(self, name: str) -> dict:
        r = self.get(name)
        open_tasks = db.rows(self.con, "SELECT id, status, request FROM repo_tasks WHERE repo_id=? AND status NOT IN "
                                       "('merged', 'discarded') ORDER BY id DESC", r["id"])
        return {**{k: r[k] for k in ("id", "name", "url", "path", "venv_python", "default_branch", "entry",
                                     "suite_status", "suite_error", "env_status", "env_error", "updated_at")},
                "known_failures": json.loads(r["known_failures"] or "[]"),
                "last_pull": json.loads(r["last_pull"] or "null"),
                "last_main_run": _brief(self._last_main_run(r)), "open_tasks": open_tasks}

    def list_repos(self) -> list[dict]:
        return [self.view(x["name"]) for x in db.rows(self.con, "SELECT name FROM repos ORDER BY name")]

    def task(self, tid: int) -> dict:
        t = self._task_row(tid)
        out = dict(t)
        for k in ("regressions", "new_tests", "fixed_tests", "new_failing"):
            out[k] = json.loads(t[k] or "[]")
        out["merge_check"] = json.loads(t["merge_check"] or "null")
        out["repo"] = self._row(t["repo_id"])["name"]
        out["before"] = _brief(self._load_run(t["before_run"]))
        out["after"] = _brief(self._load_run(t["after_run"]))
        return out

    async def task_detail(self, tid: int) -> dict:
        t = self.task(tid)
        r = self._row(t["repo_id"])
        t["diff"] = ""
        if t["branch"] and t["status"] not in ("discarded",):
            code, out = await self._exec(["git", "diff", f"{r['default_branch']}...{t['branch']}"], r["path"])
            t["diff"] = out[:60_000] if code == 0 else ""
        return t

    def tasks(self, name: str | None = None) -> list[dict]:
        rows = db.rows(self.con, "SELECT id FROM repo_tasks WHERE repo_id=? ORDER BY id DESC LIMIT 50",
                       self.get(name)["id"]) if name else db.rows(self.con, "SELECT id FROM repo_tasks ORDER BY id DESC LIMIT 50")
        return [self.task(x["id"]) for x in rows]

    def runs(self, name: str, limit: int = 20) -> list[dict]:
        r = self.get(name)
        return [_brief(x) for x in db.rows(self.con, "SELECT * FROM test_runs WHERE repo_id=? ORDER BY id DESC LIMIT ?",
                                           r["id"], limit)]

    # ---------- repos ----------
    def _project(self, name: str, path: Path) -> dict:
        p = db.one(self.con, "SELECT * FROM projects WHERE path=?", str(path))
        if p:
            return p
        pname = f"{name}-repo" if db.one(self.con, "SELECT 1 FROM projects WHERE name=?", name) else name
        pid = db.insert(self.con, "projects", name=pname, path=str(path), goal=f"Workspace clone of {name}",
                        created_at=db.now(), last_activity_at=db.now())
        return db.one(self.con, "SELECT * FROM projects WHERE id=?", pid)

    async def add_repo(self, url: str, venv=None, name: str | None = None, entry: str = "main") -> dict:
        url = (url or "").strip()
        if not URL_RE.match(url):
            raise RepoError("url must be https://…, git@… or a local path")
        name = name or re.sub(r"\.git$", "", url.rstrip("/").split("/")[-1].split(":")[-1])
        if not re.fullmatch(r"[\w.-]+", name) or name.startswith("."):
            raise RepoError(f"bad repository name {name!r}")
        if not re.fullmatch(r"[\w.]+", entry or ""):
            raise RepoError(f"bad entry module {entry!r}")
        python = self._python(venv)
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root) or path == self.root:
            raise RepoError("repository path must be inside the workspace")
        if path.exists():
            code, origin = await self._exec(["git", "remote", "get-url", "origin"], path)
            norm = lambda u: u.strip().rstrip("/").removesuffix(".git")
            if code != 0 or norm(origin) != norm(url):
                raise RepoError(f"{path} already exists and is not a clone of {url}")
        else:
            code, out = await self._exec(["git", "clone", "--quiet", url, str(path)], self.root, timeout=900)
            if code != 0:
                raise RepoError(f"git clone failed: {out.strip()[-1500:]}")
        exclude = path / ".git" / "info" / "exclude"
        with contextlib.suppress(OSError):
            text = exclude.read_text() if exclude.exists() else ""
            if "__pycache__" not in text:
                exclude.parent.mkdir(parents=True, exist_ok=True)
                exclude.write_text(text + "\n__pycache__/\n.pytest_cache/\n")
        code, head = await self._exec(["git", "symbolic-ref", "--short", "refs/remotes/origin/HEAD"], path)
        default = head.strip().split("/", 1)[1] if code == 0 and "/" in head else \
            await self.git(path, "rev-parse", "--abbrev-ref", "HEAD")
        project = self._project(name, path)
        fields = dict(url=url, path=str(path), venv_python=python, default_branch=default, entry=entry,
                      project_id=project["id"], updated_at=db.now())
        existing = db.one(self.con, "SELECT id FROM repos WHERE name=?", name)
        if existing:
            db.update(self.con, "repos", existing["id"], **fields)
        else:
            db.insert(self.con, "repos", name=name, suite_status="none", known_failures="[]", created_at=db.now(),
                      **fields)
        await self.env_check(name)
        return self.view(name)

    async def env_check(self, name: str) -> dict:
        r = self.get(name)
        code, out = await self._exec([r["venv_python"], "-c", f"import sys; sys.path.insert(0, '.'); import {r['entry']}"],
                                     r["path"], timeout=180, env=self._test_env(r, None))
        ok = code == 0
        err = None if ok else out.strip()[-2000:]
        db.update(self.con, "repos", r["id"], env_status="ok" if ok else "error", env_error=err, updated_at=db.now())
        return {"ok": ok, "error": err}

    # ---------- tests ----------
    async def _run_suite(self, r: dict, kind: str, task_id: int | None = None) -> dict:
        path = Path(r["path"])
        if not (path / SUITE_DIR).is_dir():
            raise RepoError("this repository has no assistant test suite yet (build it first)")
        overlay = await self._overlay(r["venv_python"])
        sha = await self.git(path, "rev-parse", "HEAD")
        ref = await self.git(path, "rev-parse", "--abbrev-ref", "HEAD")
        runs_dir = self.root / ".runs"
        runs_dir.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="junit", dir=runs_dir) as tmp:
            xml = Path(tmp) / "report.xml"
            start = time.monotonic()
            code, out = await self._exec([r["venv_python"], "-m", "pytest", SUITE_DIR, "-q", "-p", "no:cacheprovider",
                                          f"--junitxml={xml}", "-o", "junit_family=xunit2"],
                                         path, timeout=self.test_timeout, env=self._test_env(r, overlay))
            duration = round(time.monotonic() - start, 1)
            results: dict[str, str] = {}
            if xml.exists():
                with contextlib.suppress(ET.ParseError):
                    results = parse_junit(xml)
        counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
        for o in results.values():
            counts[{"passed": "passed", "failed": "failed", "error": "errors", "skipped": "skipped"}[o]] += 1
        if not results:  # timeout or collection crash: fall back to the summary line
            for k, pat in (("passed", r"(\d+) passed"), ("failed", r"(\d+) failed"), ("errors", r"(\d+) errors?")):
                m = re.search(pat, out)
                counts[k] = int(m.group(1)) if m else 0
            if code == 124 or (code not in (0, 5) and not any(counts.values())):
                counts["errors"] = max(counts["errors"], 1)
        rid = db.insert(self.con, "test_runs", repo_id=r["id"], task_id=task_id, ref=ref, sha=sha, kind=kind,
                        results=json.dumps(results), duration_s=duration, output_tail=out[-4000:],
                        created_at=db.now(), **counts)
        return {"id": rid, "ref": ref, "sha": sha, "kind": kind, "results": results, "duration_s": duration,
                "output_tail": out[-4000:], "timed_out": code == 124, "created_at": db.now(), **counts}

    @staticmethod
    def _failures(run: dict, names: list[str] | None = None) -> str:
        bad = names if names is not None else [t for t, o in run["results"].items() if o in ("failed", "error")]
        return "\n".join(f"- {t}" for t in bad) + "\n\nOutput tail:\n" + (run.get("output_tail") or "")[-3000:]

    async def run_tests(self, name: str, ref: str | None = None) -> dict:
        r = self.get(name)
        lock = self._lock(r["id"])
        if lock.locked():
            raise RepoError("busy: another operation is running on this repository")
        async with lock:
            path = Path(r["path"])
            current = await self.git(path, "rev-parse", "--abbrev-ref", "HEAD")
            target = ref or r["default_branch"]
            if target != current:
                if not re.fullmatch(r"[\w./-]+", target) or target.startswith("-"):
                    raise RepoError(f"bad ref {target!r}")
                await self._clean(path)
                await self.git(path, "checkout", "-q", target)
            try:
                run = await self._run_suite(r, "manual")
            finally:
                if target != current:
                    with contextlib.suppress(Exception):
                        await self.git(path, "checkout", "-q", current)
        return run

    # ---------- suite ----------
    async def build_suite(self, name: str, focus: str | None = None) -> dict:
        self._need_runner()
        r = self.get(name)
        if r["suite_status"] == "building" or self._lock(r["id"]).locked():
            raise RepoError("busy: another operation is running on this repository")
        db.update(self.con, "repos", r["id"], suite_status="building", suite_error=None, updated_at=db.now())
        self._spawn(self._build_suite(r["id"], focus))
        return self.view(name)

    async def _build_suite(self, rid: int, focus: str | None) -> None:
        r = self._row(rid)
        path = Path(r["path"])
        async with self._lock(rid):
            try:
                await self._clean(path)
                await self.git(path, "checkout", "-q", r["default_branch"])
                await self.git(path, "checkout", "-q", "-B", SUITE_BRANCH)
                cmd = self._cmd(r, await self._overlay(r["venv_python"]))
                project = self._project_row(r)
                prompt = SUITE_PROMPT.format(entry=r["entry"], cmd=cmd, focus=f"Focus especially on: {focus}" if focus else "")
                run = None
                for attempt in range(self.claude_attempts):
                    res = await self.runner.run(project, prompt, follow_up=attempt > 0)
                    if res.get("final"):
                        raise RepoError(res.get("result") or "Claude stopped")
                    await self._commit_all(path, "assistant: test suite")
                    run = await self._run_suite(r, "baseline")
                    if run["passed"] and not (run["failed"] or run["errors"]):
                        break
                    prompt = FIX_SUITE_PROMPT.format(failures=self._failures(run), cmd=cmd)
                if not run or not run["passed"]:
                    raise RepoError("the suite has no passing tests")
                known = sorted(t for t, o in run["results"].items() if o in ("failed", "error"))
                db.update(self.con, "repos", rid, suite_status="proposed", suite_branch=SUITE_BRANCH,
                          known_failures=json.dumps(known), updated_at=db.now())
            except Exception as e:
                log.exception("suite build for repo %s failed", rid)
                db.update(self.con, "repos", rid, suite_status="failed", suite_error=str(e)[:2000], updated_at=db.now())
            finally:
                with contextlib.suppress(Exception):
                    await self._back_to_default(r)

    async def accept_suite(self, name: str) -> dict:
        r = self.get(name)
        if r["suite_status"] != "proposed":
            raise RepoError(f"the suite is {r['suite_status']}, not proposed")
        async with self._lock(r["id"]):
            path = Path(r["path"])
            await self._clean(path)
            await self.git(path, "checkout", "-q", r["default_branch"])
            await self.git(path, *(await self._ident(path)), "merge", "--no-ff", "-q", "-m",
                           "assistant: add test suite", SUITE_BRANCH)
            db.update(self.con, "repos", r["id"], suite_status="accepted", updated_at=db.now())
            run = await self._run_suite(self._row(r["id"]), "baseline")
        return {"repo": name, "baseline": _brief(run)}

    # ---------- tasks ----------
    async def start_task(self, name: str, request: str) -> dict:
        self._need_runner()
        r = self.get(name)
        request = (request or "").strip()
        if not request:
            raise RepoError("describe the change")
        if r["suite_status"] != "accepted":
            raise RepoError("build and accept the test suite first")
        if db.one(self.con, f"SELECT id FROM repo_tasks WHERE repo_id=? AND status IN {ACTIVE}", r["id"]) \
                or self._lock(r["id"]).locked():
            raise RepoError("another task or operation is running on this repository")
        await self._clean(Path(r["path"]))
        tid = db.insert(self.con, "repo_tasks", repo_id=r["id"], request=request, status="queued",
                        created_at=db.now(), updated_at=db.now())
        db.update(self.con, "repo_tasks", tid, branch=f"task/{tid}-{_slug(request)}")
        self._spawn(self._task_flow(tid))
        return self.task(tid)

    def _set(self, tid: int, **fields) -> None:
        db.update(self.con, "repo_tasks", tid, updated_at=db.now(), **fields)

    async def _freeze(self, r: dict) -> str | None:
        code, out = await self._exec([r["venv_python"], "-m", "pip", "freeze"], r["path"], timeout=120)
        return out if code == 0 else None

    async def _task_flow(self, tid: int) -> None:
        t = self._task_row(tid)
        r = self._row(t["repo_id"])
        path = Path(r["path"])
        async with self._lock(r["id"]):
            try:
                await self._clean(path)
                await self.git(path, "checkout", "-q", r["default_branch"])
                base = await self.git(path, "rev-parse", "HEAD")
                await self.git(path, "checkout", "-q", "-b", t["branch"])
                freeze = await self._freeze(r)
                self._set(tid, status="testing_before", base_sha=base)
                before = await self._run_suite(r, "before", tid)
                self._set(tid, before_run=before["id"], status="claude")
                cmd = self._cmd(r, await self._overlay(r["venv_python"]))
                res = await self.runner.run(self._project_row(r),
                                            TASK_PROMPT.format(request=t["request"], branch=t["branch"], cmd=cmd),
                                            follow_up=False)
                self._set(tid, claude_session=res.get("session"), claude_result=str(res.get("result", ""))[:4000])
                if res.get("limit"):
                    self._set(tid, status="paused_limit", error=res.get("result"))
                    return
                if res.get("final"):
                    raise RepoError(res.get("result") or "Claude stopped")
                await self._finish_task(tid, r, before)
                pkg = NEEDS_RE.search(str(res.get("result") or ""))
                if pkg:
                    action = self.create_action("repo_install", r["name"], {"package": pkg.group(1), "task": tid})
                    self._set(tid, status="needs_package",
                              error=f"needs package {pkg.group(1)} (confirm action #{action['action_id']} to install it)")
                if freeze and freeze != await self._freeze(r):
                    self._set(tid, env_warning="the venv's installed packages changed during this task")
            except Exception as e:
                log.exception("task %s failed", tid)
                self._set(tid, status="failed", error=str(e)[:2000])
            finally:
                with contextlib.suppress(Exception):
                    await self._back_to_default(r)

    async def _finish_task(self, tid: int, r: dict, before: dict) -> None:
        path = Path(r["path"])
        await self._commit_all(path, f"assistant: task #{tid}")
        self._set(tid, status="testing_after")
        after = await self._run_suite(r, "after", tid)
        if after["timed_out"]:
            self._set(tid, after_run=after["id"], status="failed",
                      error=f"the test suite timed out after {self.test_timeout:g}s")
            return
        cmp = compare_runs(before["results"], after["results"], json.loads(r["known_failures"] or "[]"))
        summary = await self.runner.run(self._project_row(r), SUMMARY_PROMPT.format(base=r["default_branch"]),
                                        follow_up=True)
        self._set(tid, after_run=after["id"], status="ready", error=None,
                  head_sha=await self.git(path, "rev-parse", "HEAD"),
                  diff_stat=await self.git(path, "diff", "--stat", f"{r['default_branch']}...HEAD"),
                  summary="" if summary.get("final") else str(summary.get("result", ""))[:4000],
                  **{k: json.dumps(v) for k, v in cmp.items()})

    async def fix_task(self, tid: int) -> dict:
        self._need_runner()
        t = self._task_row(tid)
        r = self._row(t["repo_id"])
        if t["status"] not in ("ready", "failed") or not t["before_run"]:
            raise RepoError(f"task {tid} is {t['status']}; nothing to fix")
        if self._lock(r["id"]).locked():
            raise RepoError("another task or operation is running on this repository")
        self._set(tid, status="claude")
        self._spawn(self._fix_flow(tid))
        return self.task(tid)

    async def _fix_flow(self, tid: int) -> None:
        t = self._task_row(tid)
        r = self._row(t["repo_id"])
        path = Path(r["path"])
        async with self._lock(r["id"]):
            try:
                await self._clean(path)
                await self.git(path, "checkout", "-q", t["branch"])
                before = self._load_run(t["before_run"])
                after = self._load_run(t["after_run"]) or {"results": {}, "output_tail": t["error"] or ""}
                names = json.loads(t["regressions"] or "[]") + json.loads(t["new_failing"] or "[]")
                cmd = self._cmd(r, await self._overlay(r["venv_python"]))
                res = await self.runner.run(self._project_row(r),
                                            FIX_TASK_PROMPT.format(failures=self._failures(after, names), cmd=cmd),
                                            follow_up=True)
                if res.get("limit"):
                    self._set(tid, status="paused_limit", error=res.get("result"))
                    return
                if res.get("final"):
                    raise RepoError(res.get("result") or "Claude stopped")
                await self._finish_task(tid, r, before)
            except Exception as e:
                log.exception("fixing task %s failed", tid)
                self._set(tid, status="failed", error=str(e)[:2000])
            finally:
                with contextlib.suppress(Exception):
                    await self._back_to_default(r)

    async def discard(self, tid: int) -> dict:
        t = self._task_row(tid)
        r = self._row(t["repo_id"])
        if t["status"] in ACTIVE or t["status"] == "merged":
            raise RepoError(f"task {tid} is {t['status']}; it can't be discarded")
        async with self._lock(r["id"]):
            path = Path(r["path"])
            if await self.git(path, "rev-parse", "--abbrev-ref", "HEAD") == t["branch"]:
                await self._clean(path)
                await self.git(path, "checkout", "-q", r["default_branch"])
            await self._exec(["git", "branch", "-D", t["branch"]], path)
            self._set(tid, status="discarded")
        return self.task(tid)

    # ---------- merge / pull / push ----------
    async def merge_task(self, tid: int) -> dict:
        t = self._task_row(tid)
        r = self._row(t["repo_id"])
        if t["status"] != "ready":
            raise RepoError(f"task {tid} is {t['status']}, not ready")
        async with self._lock(r["id"]):
            path = Path(r["path"])
            await self._clean(path)
            await self.git(path, "checkout", "-q", r["default_branch"])
            prev = await self.git(path, "rev-parse", "HEAD")
            prev_run = self._last_main_run(r)
            code, out = await self._exec(["git", *(await self._ident(path)), "merge", "--no-ff", "-q", "-m",
                                          f"Merge task #{tid}: {t['request'][:60]}", t["branch"]], path)
            if code != 0:
                await self._exec(["git", "merge", "--abort"], path)
                self._set(tid, error=f"merge conflicts with {r['default_branch']}: {out.strip()[-800:]}")
                raise RepoError(f"task {tid} conflicts with {r['default_branch']}; ask Claude to rebase it or discard it")
            merged = await self.git(path, "rev-parse", "HEAD")
            run = await self._run_suite(r, "merge", tid)
            known = json.loads(r["known_failures"] or "[]")
            after = self._load_run(t["after_run"]) or {"results": {}}
            check = {"tests": _brief(run), "vs_branch": compare_runs(after["results"], run["results"], known),
                     "vs_main": compare_runs(prev_run["results"], run["results"], known) if prev_run else None}
            self._set(tid, status="merged", merge_prev_sha=prev, merge_sha=merged, merge_check=json.dumps(check))
        return {"task": tid, "merged": merged, "check": check}

    async def undo_merge(self, name: str) -> dict:
        r = self.get(name)
        t = db.one(self.con, "SELECT * FROM repo_tasks WHERE repo_id=? AND status='merged' "
                             "ORDER BY updated_at DESC, id DESC LIMIT 1", r["id"])
        if not t:
            raise RepoError("no assistant merge to undo")
        async with self._lock(r["id"]):
            path = Path(r["path"])
            await self._clean(path)
            await self.git(path, "checkout", "-q", r["default_branch"])
            if await self.git(path, "rev-parse", "HEAD") != t["merge_sha"]:
                raise RepoError(f"{r['default_branch']} has moved since task {t['id']} was merged; not undoing")
            await self.git(path, "reset", "--hard", "-q", t["merge_prev_sha"])  # local only, guarded by merge_sha
            self._set(t["id"], status="ready", merge_sha=None, merge_prev_sha=None, merge_check=None)
        return {"task": t["id"], "reset_to": t["merge_prev_sha"]}

    async def pull(self, name: str) -> dict:
        r = self.get(name)
        async with self._lock(r["id"]):
            path = Path(r["path"])
            await self._clean(path)
            await self.git(path, "checkout", "-q", r["default_branch"])
            old = await self.git(path, "rev-parse", "HEAD")
            prev_run = self._last_main_run(r)
            # local main may contain the accepted suite, so upstream is merged in (never fast-forward-only)
            code, out = await self._exec(["git", *(await self._ident(path)), "pull", "--no-rebase", "--no-edit", "-q",
                                          "origin", r["default_branch"]], path, timeout=300)
            if code != 0:
                await self._exec(["git", "merge", "--abort"], path)
                raise RepoError(f"git pull failed: {out.strip()[-1500:]}")
            new = await self.git(path, "rev-parse", "HEAD")
            result: dict = {"changed": new != old, "old": old, "new": new, "at": db.now()}
            if new != old:
                commits = await self.git(path, "log", "--oneline", f"{old}..{new}")
                run = await self._run_suite(r, "pull") if (path / SUITE_DIR).is_dir() else None
                known = json.loads(r["known_failures"] or "[]")
                cmp = compare_runs(prev_run["results"], run["results"], known) if prev_run and run else None
                summary = ""
                if self.runner is not None:
                    s = await self.runner.run(self._project_row(r), UPSTREAM_PROMPT.format(old=old[:12], new=new[:12]),
                                              False, wait=config.CLAUDE_WAIT)
                    summary = "" if s.get("final") else str(s.get("result", ""))[:4000]
                result.update(commits=commits.splitlines()[:50], tests=_brief(run),
                              regressions=cmp["regressions"] if cmp else [], summary=summary)
            db.update(self.con, "repos", r["id"], last_pull=json.dumps(result), updated_at=db.now())
        return result

    async def push(self, name: str, branch: str | None = None) -> dict:
        r = self.get(name)
        branch = branch or r["default_branch"]
        if not re.fullmatch(r"[\w./-]+", branch) or branch.startswith("-"):
            raise RepoError(f"bad branch name {branch!r}")
        async with self._lock(r["id"]):
            path = Path(r["path"])
            code, _ = await self._exec(["git", "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"], path)
            if code != 0:
                raise RepoError(f"no local branch {branch}")
            code, out = await self._exec(["git", "push", "origin", f"refs/heads/{branch}:refs/heads/{branch}"],
                                         path, timeout=120)  # never --force
            if code != 0:
                raise RepoError(f"git push failed: {out.strip()[-1500:]}")
        return {"pushed": branch, "output": out.strip()[-1000:]}

    async def install_package(self, name: str, package: str) -> dict:
        r = self.get(name)
        if not PKG_RE.match(package or ""):
            raise RepoError(f"bad package spec {package!r}")
        code, out = await self._exec([r["venv_python"], "-m", "pip", "install", package], r["path"], timeout=900)
        if code != 0:
            raise RepoError(f"pip install failed: {out.strip()[-1500:]}")
        return {"installed": package, "output": out.strip()[-1000:]}

    # ---------- pending actions ----------
    def create_action(self, kind: str, repo: str, payload: dict | None = None, created_by: str = "gemini") -> dict:
        if kind not in REPO_ACTIONS:
            raise RepoError(f"unknown action {kind}")
        r = self.get(repo)
        payload = dict(payload or {})
        if kind == "repo_push":
            payload["branch"] = payload.get("branch") or r["default_branch"]
        if kind == "repo_install" and not PKG_RE.match(payload.get("package") or ""):
            raise RepoError("bad package spec")
        if kind == "repo_merge":
            self._task_row(int(payload["task"]))
        summary = {"repo_merge": f"Merge task #{payload.get('task')} into {repo}'s {r['default_branch']} (local) and re-run the tests",
                   "repo_undo_merge": f"Undo the last assistant merge in {repo}",
                   "repo_push": f"Push branch {payload.get('branch')} of {repo} to origin",
                   "repo_install": f"pip install {payload.get('package')} into {r['venv_python']}"}[kind]
        aid = db.insert(self.con, "pending_actions", kind=kind, service=repo, payload=json.dumps(payload),
                        created_by=created_by, status="pending", created_at=db.now())
        return {"needs_confirmation": True, "action_id": aid, "kind": kind, "service": repo, "summary": summary}

    async def confirm_action(self, aid: int) -> dict:
        row = claim_action(self.con, aid)
        p = row["payload"]
        try:
            if row["kind"] == "repo_merge":
                res = await self.merge_task(int(p["task"]))
            elif row["kind"] == "repo_undo_merge":
                res = await self.undo_merge(row["service"])
            elif row["kind"] == "repo_push":
                res = await self.push(row["service"], p.get("branch"))
            elif row["kind"] == "repo_install":
                res = await self.install_package(row["service"], p.get("package"))
            else:
                raise RepoError(f"{row['kind']} is not a repo action")
        except Exception as e:
            finish_action(self.con, aid, error=e)
            raise
        finish_action(self.con, aid, result=res)
        return res
```

- [ ] **Step 6: Run tests** — the Task 1 selection, then `uv run pytest -q test_core.py -W error` twice → all pass, pristine.

- [ ] **Step 7: Commit**

```bash
git add workspace.py services.py db.py test_core.py
git commit -m "feat: repo workspace core (clone, claude-built suite, tasks, merge/pull checks, confirmed push)"
```

---

### Task 2: Tools, agent, API

**Files:** Modify `tools.py`, `agent.py`, `main.py`; test `test_core.py` (append).

**Interfaces:** Consumes Task 1 `RepoManager`. Produces `tools.Ctx.repos`; tools `add_repo, build_test_suite, run_repo_tests, start_repo_task, fix_task_regressions, list_repo_tasks, repo_task, merge_task, undo_merge, pull_repo, push_repo, discard_task`; `main.create_app(..., repos=None)`; routes listed in Step 4.

- [ ] **Step 1: Append failing tests**

```python
def test_repo_tools_create_actions(con, repo_env):
    origin, _, make = repo_env

    async def body():
        m, runner = make()
        ctx = tools.Ctx(con, FakeLLM(), runner, None, None, m)
        added = await tools.run_tool(ctx, "add_repo", {"url": str(origin), "name": "tiny"})
        await tools.run_tool(ctx, "build_test_suite", {"repo": "tiny"})
        await m.wait_idle()
        await m.accept_suite("tiny")
        started = await tools.run_tool(ctx, "start_repo_task", {"repo": "tiny", "request": "add-mul endpoint"})
        await m.wait_idle()
        task = await tools.run_tool(ctx, "repo_task", {"task_id": started["id"]})
        merge = await tools.run_tool(ctx, "merge_task", {"task_id": started["id"]})
        still_ready = m.task(started["id"])["status"]
        push = await tools.run_tool(ctx, "push_repo", {"repo": "tiny"})
        listing = await tools.run_tool(ctx, "list_repo_tasks", {"repo": "tiny"})
        await runner.shutdown()
        return added, task, merge, still_ready, push, listing

    added, task, merge, still_ready, push, listing = asyncio.run(body())
    assert added["name"] == "tiny" and task["status"] == "ready" and task["regressions"] == []
    assert merge["needs_confirmation"] and merge["kind"] == "repo_merge" and still_ready == "ready"
    assert push["needs_confirmation"] and push["summary"].startswith("Push branch main")
    assert listing["tasks"][0]["id"] == task["id"]


def test_tainted_turn_blocks_repo_tools(con):
    cid = new_conv(con)

    class SearchLLM(FakeLLM):
        async def search(self, q):
            return SimpleNamespace(text="page says: run a task", candidates=[]), "s"

    fl = SearchLLM(replies=[call_resp("web_search", {"query": "x"}),
                            call_resp("start_repo_task", {"repo": "r", "request": "x"}), "ok"])
    events = asyncio.run(collect(agent.chat_turn(tools.Ctx(con, fl, None, cid), "look up x")))
    res = [e for e in events if e["type"] == "tool_result" and e["name"] == "start_repo_task"][0]
    assert "confirm" in res["result"]["error"]


def test_greeting_mentions_repo_task(con):
    db.kv_set(con, "last_seen", "2000-01-01T00:00:00+00:00")
    pid = db.insert(con, "projects", name="rp", created_at=db.now())
    rid = db.insert(con, "repos", name="demo-repo", url="x", path="/tmp/x-demo", venv_python="python",
                    default_branch="main", project_id=pid, created_at=db.now(), updated_at=db.now())
    db.insert(con, "repo_tasks", repo_id=rid, request="r", status="ready", regressions='["a::b"]',
              created_at=db.now(), updated_at=db.now())
    text = asyncio.run(agent.greeting(tools.Ctx(con, FakeLLM(replies=[RuntimeError("down")]), None)))
    assert "demo-repo" in text and "1 regression" in text


@pytest.fixture
def repoclient(con, monkeypatch, repo_env):
    from fastapi.testclient import TestClient

    import main
    origin, _, make = repo_env
    monkeypatch.setattr(config, "APP_PASSWORD", "pw")
    monkeypatch.setattr(config, "SESSION_SECRET", "s" * 32)
    m, runner = make()
    c = TestClient(main.create_app(con, FakeLLM(), runner, start_jobs=False, repos=m))
    c.post("/login", json={"password": "pw"})
    return c, origin, m


def test_repos_api(repoclient, con):
    c, origin, m = repoclient
    assert c.post("/api/repos", json={"url": str(origin), "name": "tiny"}).json()["name"] == "tiny"
    assert c.post("/api/repos", json={"url": 5}).status_code == 400
    assert c.post("/api/repos/tiny/suite", json={}).status_code == 200
    for _ in range(100):
        if c.get("/api/repos/tiny").json()["suite_status"] != "building":
            break
        time.sleep(0.2)
    assert c.post("/api/repos/tiny/suite/accept").json()["baseline"]["passed"] == 2
    t = c.post("/api/repos/tiny/tasks", json={"request": "add-mul endpoint"}).json()
    for _ in range(150):
        state = c.get(f"/api/tasks/{t['id']}").json()
        if state["status"] not in ("queued", "testing_before", "claude", "testing_after"):
            break
        time.sleep(0.2)
    assert state["status"] == "ready" and "/mul" in state["diff"]
    assert c.post(f"/api/tasks/{t['id']}/merge").json()["check"]["tests"]["passed"] == 3
    assert c.get("/api/repos/tiny").json()["last_main_run"]["passed"] == 3
    assert c.post("/api/repos/ghost/pull").status_code == 400
    aid = m.create_action("repo_undo_merge", "tiny")["action_id"]
    assert c.post(f"/api/actions/{aid}/confirm").json()["task"] == t["id"]


def test_repos_api_requires_login(client):
    assert client.get("/api/repos").status_code == 401
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest -q test_core.py -k "repo_tools or blocks_repo or repo_task or repos_api" -W error` → FAIL.

- [ ] **Step 3: `tools.py`**

(a) `Ctx` gains a last field `repos: object = None`.

(b) Append:
```python
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
    t = _repos(ctx).task(int(task_id))
    if t["status"] != "ready":
        raise ToolError(f"task {task_id} is {t['status']}, not ready to merge")
    return _repos(ctx).create_action("repo_merge", t["repo"], {"task": int(task_id)})


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
```

- [ ] **Step 4: `agent.py`**

(a) Add to `GUARDED_TOOLS`: `"add_repo", "build_test_suite", "start_repo_task", "fix_task_regressions", "pull_repo"`.
(b) Add to the taint tuple (`tainted = tainted or call.name in (...)`): `"repo_task", "list_repo_tasks", "pull_repo"`.
(c) In `greeting`, after the `down = [...]` statement:
```python
    repo_done = db.rows(con, "SELECT t.id, r.name, t.status, t.regressions FROM repo_tasks t JOIN repos r "
                             "ON r.id = t.repo_id WHERE t.status IN ('ready', 'failed', 'needs_package') "
                             "AND t.updated_at > ?", last_seen) if last_seen else []
    upstream = [(x["name"], json.loads(x["last_pull"])) for x in
                db.rows(con, "SELECT name, last_pull FROM repos WHERE last_pull IS NOT NULL")]
```
and in `facts`, before `"Services that should be running but are down:",`:
```python
        "Repo tasks finished since last visit:",
        "\n".join(f"- #{t['id']} {t['name']}: {t['status']}, {len(json.loads(t['regressions'] or '[]'))} regression(s)"
                  for t in repo_done) or "(none)", "",
        "Upstream regressions found by the last pull:",
        "\n".join(f"- {n}: {', '.join(p.get('regressions') or [])}" for n, p in upstream
                  if p and p.get("regressions")) or "(none)", "",
```

- [ ] **Step 5: `main.py`**

(a) `from workspace import RepoManager` ; `create_app(con, llm, runner, start_jobs: bool = True, services=None, repos=None)`; `ctx()` returns `tools.Ctx(con, llm, runner, cid, services, repos)`.
(b) Replace the confirm route:
```python
    @app.post("/api/actions/{aid}/confirm")
    async def actions_confirm(aid: int):
        row = db.one(con, "SELECT kind FROM pending_actions WHERE id=?", aid)
        if row and row["kind"].startswith("repo_"):
            return await svc_call(repos_or_503().confirm_action(aid))
        return await svc_call(svc_or_503().confirm_action(aid))
```
and make `/api/actions` and `/api/actions/{aid}/cancel` work without the service manager:
```python
    @app.get("/api/actions")
    async def actions_list():
        cutoff = db.to_utc(datetime.now(timezone.utc) - timedelta(minutes=15))
        return db.rows(con, "SELECT * FROM pending_actions WHERE status='pending' AND created_at >= ? ORDER BY id DESC", cutoff)

    @app.post("/api/actions/{aid}/cancel")
    async def actions_cancel(aid: int):
        con.execute("UPDATE pending_actions SET status='cancelled', decided_at=? WHERE id=? AND status='pending'",
                    (db.now(), aid))
        return {"ok": True}
```
(import `datetime, timedelta, timezone` if not already imported).
(c) Routes before `return app` (uses existing `svc_call`, which maps `ServiceError` — `RepoError` subclasses it — to 400):
```python
    # ---------- repos ----------

    def repos_or_503() -> RepoManager:
        if repos is None:
            raise HTTPException(503, "The repo workspace is not available")
        return repos

    def opt_str(body: dict, key: str) -> str | None:
        v = body.get(key)
        if v is not None and not isinstance(v, str):
            raise HTTPException(400, f"{key} must be a string")
        return v

    def sync(fn, *a):
        try:
            return fn(*a)
        except ServiceError as e:
            raise HTTPException(400, str(e))

    @app.get("/api/repos")
    async def repos_list():
        return repos_or_503().list_repos()

    @app.post("/api/repos")
    async def repos_add(body: dict):
        url = opt_str(body, "url")
        if not url:
            raise HTTPException(400, "url is required")
        return await svc_call(repos_or_503().add_repo(url, opt_str(body, "venv"), opt_str(body, "name"),
                                                      opt_str(body, "entry") or "main"))

    @app.get("/api/repos/{name}")
    async def repos_get(name: str):
        return {**sync(repos_or_503().view, name), "runs": sync(repos_or_503().runs, name)}

    @app.post("/api/repos/{name}/env-check")
    async def repos_env_check(name: str):
        return await svc_call(repos_or_503().env_check(name))

    @app.post("/api/repos/{name}/suite")
    async def repos_suite(name: str, body: dict = Body(default={})):
        return await svc_call(repos_or_503().build_suite(name, opt_str(body, "focus")))

    @app.post("/api/repos/{name}/suite/accept")
    async def repos_suite_accept(name: str):
        return await svc_call(repos_or_503().accept_suite(name))

    @app.post("/api/repos/{name}/test")
    async def repos_test(name: str, body: dict = Body(default={})):
        run = await svc_call(repos_or_503().run_tests(name, opt_str(body, "ref")))
        return {k: v for k, v in run.items() if k != "results"} | \
            {"failing": [t for t, o in run["results"].items() if o in ("failed", "error")]}

    @app.post("/api/repos/{name}/tasks")
    async def repos_task_start(name: str, body: dict):
        return await svc_call(repos_or_503().start_task(name, opt_str(body, "request") or ""))

    @app.get("/api/repos/{name}/tasks")
    async def repos_tasks(name: str):
        return sync(repos_or_503().tasks, name)

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
        return await svc_call(repos_or_503().push(name, opt_str(body, "branch")))
```
Note: `svc_call` wraps coroutines; `repos_or_503()` raising `HTTPException` before the coroutine is created is fine. If the existing `svc_call` only catches `ServiceError`, `RepoError` is covered by inheritance.
(d) `main()`: after `services = ServiceManager(...)`:
```python
    repos = RepoManager(con, runner=runner, test_timeout=config.REPO_TEST_TIMEOUT)
```
and pass `repos=repos` to `create_app`. Add to `config.py` DEFAULTS `"REPO_TEST_TIMEOUT": "300"` and constant `REPO_TEST_TIMEOUT = int(_env["REPO_TEST_TIMEOUT"])`.

- [ ] **Step 6: Run tests** — `uv run pytest -q test_core.py -W error` twice → all pass, pristine.

- [ ] **Step 7: Commit**

```bash
git add tools.py agent.py main.py config.py test_core.py
git commit -m "feat: repo workspace tools, guard, greeting and API"
```

---

### Task 3: Repos tab UI + README

**Files:** Modify `static/index.html`, `README.md`.

- [ ] **Step 1: HTML**

(a) After the closing `</section>` of `<section id="services">`:
```html
  <section id="repos">
    <div class="row"><h2>Repos</h2><button class="ghost" onclick="addRepo()">＋ Repo</button></div>
    <div id="rlist2"></div>
  </section>
```
(b) In `<nav>`, after the Services button: `<button data-v="repos" onclick="show('repos')">🐙 Repos</button>`
(c) After the `ssheet` div:
```html
<div class="sheet" id="rsheet"><div>
  <div class="row"><b id="rtitle"></b><span class="chip" id="rstate"></span>
    <button class="ghost" onclick="closeRepo()">✕ Close</button></div>
  <div class="row" style="margin:8px 0">
    <button class="ghost" onclick="repoAct('test')">▶ Run tests</button>
    <button class="ghost" onclick="repoAct('pull')">⤓ Pull</button>
    <button class="ghost" onclick="repoAct('push')">⤒ Push main</button>
    <button class="ghost" onclick="repoAct('undo-merge')">↶ Undo merge</button>
  </div>
  <div id="rinfo"></div>
  <form class="send" onsubmit="newRepoTask(event)"><input id="rtask" placeholder="Describe a change for Claude…"><button class="btn">Start task</button></form>
  <h2>Test suite</h2><div id="rsuite"></div>
  <h2>Tasks</h2><div id="rtasks"></div>
  <h2>Test runs</h2><div id="rruns"></div>
</div></div>
```

- [ ] **Step 2: JS**

(a) `loaders` gains `repos: loadRepos` (after `services: loadServices`).
(b) Add before `// ---------- status ----------`:
```js
// ---------- repos ----------
let repoCur = null, repoTimer = null;
const runTxt = r => r ? `${r.passed}✓ ${r.failed + r.errors}✗${r.timed_out ? ' (timed out)' : ''} · ${fmt(r.created_at)}` : 'no runs yet';
const listTxt = xs => (xs || []).map(x => `<div class="small" style="font-family:ui-monospace,monospace">${esc(x)}</div>`).join('');
function repoCard(r) {
  return `<div class="card click" data-repo="${esc(r.name)}">
    <div class="row"><b>${esc(r.name)}</b><span class="st">suite: ${esc(r.suite_status)}</span>${r.env_status === 'error' ? '<span class="st" style="color:var(--warn)">env error</span>' : ''}</div>
    <div class="muted small">${esc(r.default_branch)} · ${runTxt(r.last_main_run)} · ${r.open_tasks.length} open task(s)</div></div>`;
}
async function loadRepos() {
  const rs = await api('/api/repos');
  $('#rlist2').innerHTML = rs.map(repoCard).join('') || '<p class="muted">No repos yet. Tap ＋ Repo or tell the assistant a GitHub URL.</p>';
}
document.addEventListener('click', e => { const c = e.target.closest('[data-repo]'); if (c) openRepo(c.dataset.repo); });
async function addRepo() {
  const url = prompt('Git URL to clone (https://github.com/…)');
  if (!url) return;
  const venv = prompt('Existing venv that can run the app (folder or python path, optional)') || undefined;
  toast('Cloning…', true);
  try { await api('/api/repos', {method: 'POST', body: {url, venv}}); } catch { return; }
  loadRepos();
}
async function openRepo(name) {
  repoCur = name;
  $('#rtitle').textContent = name;
  $('#rinfo').innerHTML = '';
  $('#rsheet').classList.add('on');
  await refreshRepo();
  clearInterval(repoTimer);
  repoTimer = setInterval(() => refreshRepo().catch(() => {}), 3000);
}
function closeRepo() { clearInterval(repoTimer); repoCur = null; $('#rsheet').classList.remove('on'); loadRepos(); }
function taskCard(t) {
  const id = Number(t.id);
  const busy = ['queued', 'testing_before', 'claude', 'testing_after'].includes(t.status);
  const regs = (t.regressions || []).length;
  const btns = [];
  if (t.claude_session) btns.push(`<button class="ghost" onclick="closeRepo();show('claude');openSession(${Number(t.claude_session)})">Watch</button>`);
  if (t.status === 'ready') btns.push(`<button class="btn" onclick="repoTaskAct(${id}, 'merge')">Merge to main</button>`);
  if (t.status === 'ready' && regs + (t.new_failing || []).length) btns.push(`<button class="ghost" onclick="repoTaskAct(${id}, 'fix')">Fix regressions</button>`);
  if (!busy && !['merged', 'discarded'].includes(t.status)) btns.push(`<button class="ghost" onclick="repoTaskAct(${id}, 'discard')">Discard</button>`);
  return `<div class="card"><div class="row"><b>#${id} ${esc(t.request)}</b><span class="st">${esc(t.status)}${busy ? ' …' : ''}</span></div>
    <div class="muted small">${esc(t.branch || '')} · before ${runTxt(t.before)} → after ${runTxt(t.after)}</div>
    ${regs ? `<div class="small" style="color:var(--warn)">⚠ ${regs} regression(s):</div>${listTxt(t.regressions)}` : ''}
    ${(t.new_tests || []).length ? `<div class="small">new tests:</div>${listTxt(t.new_tests)}` : ''}
    ${(t.fixed_tests || []).length ? `<div class="small">fixed:</div>${listTxt(t.fixed_tests)}` : ''}
    ${t.error ? `<div class="small" style="color:var(--warn)">${esc(t.error)}</div>` : ''}
    ${t.env_warning ? `<div class="small" style="color:var(--warn)">${esc(t.env_warning)}</div>` : ''}
    ${t.summary ? `<div class="small">${md(t.summary)}</div>` : ''}
    ${t.merge_check ? `<div class="small">merge check: ${runTxt(t.merge_check.tests)}${t.merge_check.vs_main && t.merge_check.vs_main.regressions.length ? ` · <span style="color:var(--warn)">regressions vs main: ${esc(t.merge_check.vs_main.regressions.join(', '))}</span>` : ''}</div>` : ''}
    ${t.diff_stat ? `<details><summary class="small">diff stat</summary><pre class="small">${esc(t.diff_stat)}</pre></details>` : ''}
    <div class="row" style="margin-top:6px"><span></span>${btns.join('')}</div></div>`;
}
async function refreshRepo() {
  if (!repoCur) return;
  const name = repoCur;
  const [r, tasks] = await Promise.all([api(`/api/repos/${encodeURIComponent(name)}`), api(`/api/repos/${encodeURIComponent(name)}/tasks`)]);
  if (repoCur !== name) return;
  $('#rstate').textContent = `suite: ${r.suite_status}`;
  const suiteBtns = r.suite_status === 'proposed'
    ? `<button class="btn" onclick="repoSuite('accept')">Accept suite</button> <button class="ghost" onclick="repoSuite('build')">Rebuild</button>`
    : r.suite_status === 'building' ? '<span class="muted small">Claude is writing tests…</span>'
    : `<button class="ghost" onclick="repoSuite('build')">${r.suite_status === 'accepted' ? 'Extend / rebuild' : 'Build test suite'}</button>`;
  $('#rsuite').innerHTML = `<div class="card"><div class="small">${esc(r.path)} · venv ${esc(r.venv_python)}</div>
    ${r.env_status === 'error' ? `<div class="small" style="color:var(--warn)">environment problem:</div><pre class="small">${esc(r.env_error || '')}</pre>` : ''}
    ${r.suite_error ? `<div class="small" style="color:var(--warn)">${esc(r.suite_error)}</div>` : ''}
    ${r.known_failures.length ? `<div class="small">known failures (ignored):</div>${listTxt(r.known_failures)}` : ''}
    ${r.last_pull && r.last_pull.changed ? `<div class="small">last pull: ${esc((r.last_pull.commits || []).length + ' commit(s)')}${r.last_pull.regressions && r.last_pull.regressions.length ? ` · <span style="color:var(--warn)">regressions: ${esc(r.last_pull.regressions.join(', '))}</span>` : ''}</div>${r.last_pull.summary ? `<div class="small">${md(r.last_pull.summary)}</div>` : ''}` : ''}
    <div class="row" style="margin-top:6px"><span></span>${suiteBtns}</div></div>`;
  $('#rtasks').innerHTML = tasks.map(taskCard).join('') || '<p class="muted small">No tasks yet.</p>';
  $('#rruns').innerHTML = r.runs.map(x => `<div class="small muted">${esc(x.kind)} · ${esc(x.ref)} · ${runTxt(x)}</div>`).join('') || '<p class="muted small">No runs yet.</p>';
}
async function repoSuite(what) {
  if (!repoCur) return;
  try {
    if (what === 'accept') await api(`/api/repos/${encodeURIComponent(repoCur)}/suite/accept`, {method: 'POST'});
    else await api(`/api/repos/${encodeURIComponent(repoCur)}/suite`, {method: 'POST', body: {focus: prompt('Anything to focus on? (optional)') || undefined}});
  } catch { return; }
  refreshRepo();
}
async function newRepoTask(e) {
  e.preventDefault();
  const request = $('#rtask').value.trim();
  if (!request || !repoCur) return;
  try { await api(`/api/repos/${encodeURIComponent(repoCur)}/tasks`, {method: 'POST', body: {request}}); } catch { return; }
  $('#rtask').value = '';
  refreshRepo();
}
async function repoTaskAct(id, what) {
  if (what === 'merge' && !confirm('Merge this task into main (local) and re-run the tests?')) return;
  if (what === 'discard' && !confirm('Delete this task branch?')) return;
  $('#rinfo').innerHTML = `<p class="muted small">${esc(what)}…</p>`;
  try {
    const r = await api(`/api/tasks/${id}/${what}`, {method: 'POST'});
    const regs = r.check && r.check.vs_main ? r.check.vs_main.regressions : [];
    $('#rinfo').innerHTML = regs.length ? `<p class="small" style="color:var(--warn)">Merged, but ${regs.length} test(s) regressed vs main — consider Undo merge.</p>` : '';
  } catch { $('#rinfo').innerHTML = ''; }
  refreshRepo();
}
async function repoAct(what) {
  if (!repoCur) return;
  if (what === 'push' && !confirm(`Push ${repoCur}'s main branch to origin?`)) return;
  if (what === 'undo-merge' && !confirm('Undo the last assistant merge (reset main to before it)?')) return;
  $('#rinfo').innerHTML = `<p class="muted small">${esc(what)}…</p>`;
  try {
    const r = await api(`/api/repos/${encodeURIComponent(repoCur)}/${what}`, {method: 'POST', body: {}});
    $('#rinfo').innerHTML = what === 'test' ? `<p class="small">${runTxt(r)}</p>${listTxt(r.failing)}`
      : what === 'pull' ? `<p class="small">${r.changed ? `${(r.commits || []).length} new commit(s)` : 'Already up to date'}${r.regressions && r.regressions.length ? ` · <span style="color:var(--warn)">regressions: ${esc(r.regressions.join(', '))}</span>` : ''}</p>`
      : `<p class="small">${esc(JSON.stringify(r).slice(0, 300))}</p>`;
  } catch { $('#rinfo').innerHTML = ''; }
  refreshRepo();
}
```

- [ ] **Step 3: README** — add after the "## Services (tmux)" section:
```markdown
## Repos (GitHub coder + tester)
- **🐙 Repos → ＋ Repo** (or tell the assistant a GitHub URL): clones into `workspace/<name>`; give it an existing venv that can run the app (pytest is added from `workspace/.testtools`, your venv is not modified).
- **Build test suite**: Claude writes `tests_assistant/` (fast, faked models/network) on branch `assistant/tests`; review and tap **Accept suite**.
- **Start task**: each request gets a branch, tests before, Claude's change (watch live), tests after, regressions in red and a behaviour summary. **Merge to main** (local) re-runs the tests; **Undo merge** resets main if nothing moved since.
- **Pull** merges upstream into the local main and reports regressions; **Push** never force-pushes and uses your git credentials. From chat, merge / undo / push / package installs show **Confirm** buttons.
```

- [ ] **Step 4: Verify** — `node --check` on the extracted inline script; `uv run pytest -q test_core.py -W error` green; cross-check every `/api/repos`, `/api/tasks`, `/api/actions` path + method against main.py.

- [ ] **Step 5: Commit**

```bash
git add static/index.html README.md
git commit -m "feat: repos tab with suite, tasks, merge checks and push"
```

---

### Task 4 (controller): live check

Scratch DB. `add_repo` for the existing clone `workspace/RAG_for_192_server` (URL `https://github.com/Neel-tetranetics/RAG_for_192_server.git`, venv `/mnt/newdisk/neel/Backend_rag_server/ai-rag/venv`) — note it currently has branch `assistant-test` and is on it; switch the clone to `main` first. Build the real suite with real Claude, accept, run one real task ("fix the /upload_url mismatch in test_rag_config.json" or similar small change), merge, then push `main` to the sandbox origin after Confirm. Verify the user's running services and tmux sessions are untouched and ai-rag's venv `pip freeze` is unchanged.
