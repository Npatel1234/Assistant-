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
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import config
import db
from services import ServiceError, claim_action, existing_action, finish_action

log = logging.getLogger("workspace")
ACTIVE = ("queued", "testing_before", "claude", "testing_after")
SUITE_DIR = "tests_assistant"
SUITE_BRANCH = "assistant/tests"
PKG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(==[\w.]+)?")  # use fullmatch; no leading "-" so never a pip option
NEEDS_RE = re.compile(r"NEEDS_PACKAGE:\s*([A-Za-z0-9._-]+(?:==[\w.]+)?)")
URL_RE = re.compile(r"^(https://|git@|file://|/)")
REPO_ACTIONS = {"repo_merge", "repo_undo_merge", "repo_push", "repo_install"}
BUSY = "another task or operation is running on this repository"

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
    # a test that passed before and vanished (deleted, or lost to a collection error) is a regression too
    return {"regressions": sorted(({t for t, o in after.items() if o in bad and before.get(t) == "passed"} |
                                   {t for t, o in before.items() if o == "passed" and t not in after}) - set(known)),
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
        # an (empty) pytest.ini here stops pytest's config search from climbing into the assistant's own pyproject
        stop = self.root / "pytest.ini"
        if not stop.exists():
            stop.write_text("# stops pytest config discovery for workspace repos without their own config\n")
        self.test_timeout, self.claude_attempts = test_timeout, claude_attempts
        self._locks: dict[int, asyncio.Lock] = {}
        self.busy_grace = 10.0  # seconds an operation waits for the repo lock before refusing as busy
        self._bg: set[asyncio.Task] = set()
        con.execute(f"UPDATE repo_tasks SET status='failed', error='interrupted by restart', updated_at=? "
                    f"WHERE status IN {ACTIVE}", (db.now(),))
        con.execute("UPDATE repos SET suite_status='failed', suite_error='interrupted by restart', updated_at=? "
                    "WHERE suite_status='building'", (db.now(),))
        con.execute("UPDATE pending_actions SET status='failed', result='interrupted by restart', decided_at=? "
                    "WHERE status='running' AND kind LIKE 'repo\\_%' ESCAPE '\\'", (db.now(),))

    # ---------- plumbing ----------
    def _lock(self, repo_id: int) -> asyncio.Lock:
        return self._locks.setdefault(repo_id, asyncio.Lock())

    @contextlib.asynccontextmanager
    async def _exclusive(self, repo_id: int):
        """Take the repo lock, or refuse when it stays held (never queue behind a long run). The short grace covers
        a flow that already marked its task ready and is just returning to the default branch."""
        lock = self._lock(repo_id)
        try:
            await asyncio.wait_for(lock.acquire(), timeout=self.busy_grace)
        except TimeoutError:
            raise RepoError(BUSY) from None
        try:
            yield
        finally:
            lock.release()

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
        except BaseException:  # cancellation etc.: don't leak the process group
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)
            raise
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

    async def _park(self, r: dict) -> None:
        """End of any flow that ran Claude: keep half-done work on the assistant's own branch, return to default."""
        path = Path(r["path"])
        branch = await self.git(path, "rev-parse", "--abbrev-ref", "HEAD")
        if branch.startswith("task/") or branch == SUITE_BRANCH:  # never commit on the default branch
            await self._commit_all(path, "assistant: WIP (interrupted)")
        await self._back_to_default(r)

    async def _suite_extra(self, r: dict) -> list[str]:
        """Files the suite branch changes outside tests_assistant/ (it must only add tests)."""
        out = await self.git(Path(r["path"]), "diff", "--name-only", f"{r['default_branch']}...{SUITE_BRANCH}")
        return [f for f in out.splitlines() if not f.startswith(SUITE_DIR + "/")]

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
        return f"{env} {r['venv_python']} -m pytest {SUITE_DIR} -q -p no:cacheprovider --rootdir=."

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

    async def _baseline(self, r: dict, sha: str) -> dict | None:
        """Latest run of exactly this commit; None when there is none (e.g. runs of undone merges don't count)."""
        row = db.one(self.con, "SELECT id FROM test_runs WHERE repo_id=? AND sha=? ORDER BY id DESC LIMIT 1",
                     r["id"], sha)
        return self._load_run(row["id"]) if row else None

    @staticmethod
    def _run_error(run: dict | None) -> str | None:
        if run is None:
            return None
        if run.get("timed_out"):
            return "tests timed out"
        return None if run.get("results") else "no test results"

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
            # base..head stays meaningful after the merge (main...branch is empty once merged)
            rng = f"{t['base_sha']}..{t['head_sha']}" if t.get("base_sha") and t.get("head_sha") \
                else f"{r['default_branch']}...{t['branch']}"
            code, out = await self._exec(["git", "diff", rng], r["path"])
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
                                          f"--rootdir={path}", f"--junitxml={xml}", "-o", "junit_family=xunit2"],
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
                extra = await self._suite_extra(r)
                db.update(self.con, "repos", rid, suite_status="proposed", suite_branch=SUITE_BRANCH,
                          known_failures=json.dumps(known), updated_at=db.now(),
                          suite_error=f"suite branch also changes: {', '.join(extra)}"[:2000] if extra else None)
            except Exception as e:
                log.exception("suite build for repo %s failed", rid)
                db.update(self.con, "repos", rid, suite_status="failed", suite_error=str(e)[:2000], updated_at=db.now())
            finally:
                with contextlib.suppress(Exception):
                    await self._park(r)

    async def accept_suite(self, name: str) -> dict:
        r = self.get(name)
        async with self._exclusive(r["id"]):
            r = self._row(r["id"])
            if r["suite_status"] != "proposed":
                raise RepoError(f"the suite is {r['suite_status']}, not proposed")
            path = Path(r["path"])
            await self._clean(path)
            extra = await self._suite_extra(r)  # from git, not the stored message
            if extra:
                raise RepoError(f"the suite branch also changes files outside {SUITE_DIR}/: {', '.join(extra)}")
            await self.git(path, "checkout", "-q", r["default_branch"])
            code, out = await self._exec(["git", *(await self._ident(path)), "merge", "--no-ff", "-q", "-m",
                                          "assistant: add test suite", SUITE_BRANCH], path)
            if code != 0:
                await self._exec(["git", "merge", "--abort"], path)
                raise RepoError(f"merging the suite failed: {out.strip()[-1500:]}")
            db.update(self.con, "repos", r["id"], suite_status="accepted", suite_error=None, updated_at=db.now())
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
            # a failed rebuild must not lock out an already-accepted suite (it is still on the default branch)
            code = 1
            if r["suite_status"] == "failed":
                code, _ = await self._exec(["git", "cat-file", "-e", f"{r['default_branch']}:{SUITE_DIR}"], r["path"])
            if code != 0:
                raise RepoError("build and accept the test suite first")
        if db.one(self.con, f"SELECT id FROM repo_tasks WHERE repo_id=? AND status IN {ACTIVE}", r["id"]) \
                or self._lock(r["id"]).locked():
            raise RepoError(BUSY)
        await self._clean(Path(r["path"]))
        # no await between this check and the insert, so concurrent calls can't both pass
        if db.one(self.con, f"SELECT id FROM repo_tasks WHERE repo_id=? AND status IN {ACTIVE}", r["id"]) \
                or self._lock(r["id"]).locked():
            raise RepoError(BUSY)
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
                await self._finish_task(tid, r, before, str(res.get("result") or ""))
                if freeze and freeze != await self._freeze(r):
                    self._set(tid, env_warning="the venv's installed packages changed during this task")
            except Exception as e:
                log.exception("task %s failed", tid)
                self._set(tid, status="failed", error=str(e)[:2000])
            finally:
                with contextlib.suppress(Exception):
                    await self._park(r)

    def _session_row(self, t: dict) -> dict | None:
        """The task's own Claude session, so fix/continue don't talk into whatever the project ran last."""
        return db.one(self.con, "SELECT * FROM claude_sessions WHERE id=?", t["claude_session"]) \
            if t["claude_session"] else None

    async def _finish_task(self, tid: int, r: dict, before: dict, claude_result: str = "") -> None:
        path = Path(r["path"])
        if err := self._run_error(before):
            self._set(tid, status="failed", error=f"before-run: {err}")
            return
        await self._commit_all(path, f"assistant: task #{tid}")
        self._set(tid, status="testing_after")
        after = await self._run_suite(r, "after", tid)
        if after["timed_out"]:
            self._set(tid, after_run=after["id"], status="failed",
                      error=f"the test suite timed out after {self.test_timeout:g}s")
            return
        cmp = compare_runs(before["results"], after["results"], json.loads(r["known_failures"] or "[]"))
        summary = await self.runner.run(self._project_row(r), SUMMARY_PROMPT.format(base=r["default_branch"]),
                                        follow_up=True, row=self._session_row(self._task_row(tid)))
        self._set(tid, after_run=after["id"], status="ready", error=None,
                  head_sha=await self.git(path, "rev-parse", "HEAD"),
                  diff_stat=await self.git(path, "diff", "--stat", f"{r['default_branch']}...HEAD"),
                  summary="" if summary.get("final") else str(summary.get("result", ""))[:4000],
                  **{k: json.dumps(v) for k, v in cmp.items()})
        if pkg := NEEDS_RE.search(claude_result):  # only a task that finished cleanly; a timeout stays failed
            action = self.create_action("repo_install", r["name"], {"package": pkg.group(1), "task": tid})
            self._set(tid, status="needs_package",
                      error=f"needs package {pkg.group(1)} (confirm action #{action['action_id']} to install it)")

    async def fix_task(self, tid: int) -> dict:
        """Fix regressions, or Continue a task that paused (usage limit), needs a package, or failed."""
        self._need_runner()
        t = self._task_row(tid)
        r = self._row(t["repo_id"])
        if t["status"] not in ("ready", "failed", "paused_limit", "needs_package") or not t["before_run"]:
            raise RepoError(f"task {tid} is {t['status']}; nothing to fix")
        if self._lock(r["id"]).locked():
            raise RepoError(BUSY)
        self._set(tid, status="claude")
        self._spawn(self._fix_flow(tid, t["status"]))
        return self.task(tid)

    async def _fix_flow(self, tid: int, was: str) -> None:
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
                if was == "paused_limit":
                    prompt = f"Continue the task where you stopped: {t['request']}"
                elif was == "needs_package" or (was == "failed" and not names):
                    prompt = TASK_PROMPT.format(request=t["request"], branch=t["branch"], cmd=cmd)
                    if was == "needs_package":
                        prompt += "\nThe needed package is now installed."
                else:
                    prompt = FIX_TASK_PROMPT.format(failures=self._failures(after, names), cmd=cmd)
                res = await self.runner.run(self._project_row(r), prompt, follow_up=True, row=self._session_row(t))
                self._set(tid, claude_result=str(res.get("result", ""))[:4000])
                if res.get("limit"):
                    self._set(tid, status="paused_limit", error=res.get("result"))
                    return
                if res.get("final"):
                    raise RepoError(res.get("result") or "Claude stopped")
                await self._finish_task(tid, r, before, str(res.get("result") or ""))
            except Exception as e:
                log.exception("fixing task %s failed", tid)
                self._set(tid, status="failed", error=str(e)[:2000])
            finally:
                with contextlib.suppress(Exception):
                    await self._park(r)

    async def discard(self, tid: int) -> dict:
        t = self._task_row(tid)
        r = self._row(t["repo_id"])
        if t["status"] in ACTIVE or t["status"] in ("merged", "discarded"):
            raise RepoError(f"task {tid} is {t['status']}; it can't be discarded")
        async with self._exclusive(r["id"]):
            t = self._task_row(tid)
            if t["status"] in ACTIVE or t["status"] in ("merged", "discarded"):
                raise RepoError(f"task {tid} is {t['status']}; it can't be discarded")
            path = Path(r["path"])
            if await self.git(path, "rev-parse", "--abbrev-ref", "HEAD") == t["branch"]:
                await self._commit_all(path, "assistant: WIP (discarded)")  # the branch is deleted next
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
        async with self._exclusive(r["id"]):
            t = self._task_row(tid)
            if t["status"] != "ready":
                raise RepoError(f"task {tid} is {t['status']}, not ready")
            path = Path(r["path"])
            await self._clean(path)
            await self.git(path, "checkout", "-q", r["default_branch"])
            prev = await self.git(path, "rev-parse", "HEAD")
            prev_run = await self._baseline(r, prev)
            code, out = await self._exec(["git", *(await self._ident(path)), "merge", "--no-ff", "-q", "-m",
                                          f"Merge task #{tid}: {t['request'][:60]}", t["branch"]], path)
            if code != 0:
                await self._exec(["git", "merge", "--abort"], path)
                self._set(tid, error=f"merge conflicts with {r['default_branch']}: {out.strip()[-800:]}")
                raise RepoError(f"task {tid} conflicts with {r['default_branch']}; ask Claude to rebase it or discard it")
            merged = await self.git(path, "rev-parse", "HEAD")
            self._set(tid, status="merged", merge_prev_sha=prev, merge_sha=merged)  # record before anything can fail
            known = json.loads(r["known_failures"] or "[]")
            try:
                run = await self._run_suite(r, "merge", tid)
                after = self._load_run(t["after_run"]) or {"results": {}}
                check = {"tests": _brief(run), "baseline": _brief(prev_run), "vs_branch": None, "vs_main": None}
                err = self._run_error(run)
                if err:
                    check.update(status="failed", error=err)
                elif (berr := self._run_error(prev_run) if prev_run else "no baseline run for this commit"):
                    check.update(status="no_baseline", error=berr)
                else:
                    check.update(status="ok", error=None, vs_main=compare_runs(prev_run["results"], run["results"], known),
                                 vs_branch=compare_runs(after["results"], run["results"], known) if after["results"] else None)
            except Exception as e:
                log.exception("merge check for task %s failed", tid)
                check = {"status": "failed", "error": f"the post-merge test run failed: {e}"[:500], "tests": None,
                         "baseline": None, "vs_branch": None, "vs_main": None}
            self._set(tid, merge_check=json.dumps(check))
        return {"task": tid, "merged": merged, "check": check}

    async def undo_merge(self, name: str) -> dict:
        r = self.get(name)
        t = db.one(self.con, "SELECT * FROM repo_tasks WHERE repo_id=? AND status='merged' "
                             "ORDER BY updated_at DESC, id DESC LIMIT 1", r["id"])
        if not t:
            raise RepoError("no assistant merge to undo")
        async with self._exclusive(r["id"]):
            t = db.one(self.con, "SELECT * FROM repo_tasks WHERE id=? AND status='merged'", t["id"])
            if not t:
                raise RepoError("no assistant merge to undo")
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
        async with self._exclusive(r["id"]):
            path = Path(r["path"])
            await self._clean(path)
            await self.git(path, "checkout", "-q", r["default_branch"])
            old = await self.git(path, "rev-parse", "HEAD")
            prev_run = await self._baseline(r, old)
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
                err = "no test suite" if run is None else self._run_error(run) or \
                    ("no baseline run for the previous commit" if prev_run is None else self._run_error(prev_run))
                cmp = None if err else compare_runs(prev_run["results"], run["results"], known)
                summary = ""
                if self.runner is not None:
                    s = await self.runner.run(self._project_row(r), UPSTREAM_PROMPT.format(old=old[:12], new=new[:12]),
                                              False, wait=config.CLAUDE_WAIT)
                    summary = "" if s.get("final") else str(s.get("result", ""))[:4000]
                result.update(commits=commits.splitlines()[:50], tests=_brief(run),
                              regressions=cmp["regressions"] if cmp else [], summary=summary,
                              check="ok" if cmp else ("failed" if run and (run["timed_out"] or not run["results"]) else "no_baseline"),
                              error=err)
            db.update(self.con, "repos", r["id"], last_pull=json.dumps(result), updated_at=db.now())
        return result

    async def push(self, name: str, branch: str | None = None) -> dict:
        r = self.get(name)
        branch = branch or r["default_branch"]
        if not re.fullmatch(r"[\w./-]+", branch) or branch.startswith("-"):
            raise RepoError(f"bad branch name {branch!r}")
        async with self._exclusive(r["id"]):
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
        if not PKG_RE.fullmatch(package or ""):
            raise RepoError(f"bad package spec {package!r}")
        async with self._exclusive(r["id"]):
            code, out = await self._exec([r["venv_python"], "-m", "pip", "install", "--", package], r["path"],
                                         timeout=900)
        if code != 0:
            raise RepoError(f"pip install failed: {out.strip()[-1500:]}")
        return {"installed": package, "output": out.strip()[-1000:]}

    # ---------- pending actions ----------
    def _mergeable(self, r: dict, payload: dict) -> int:
        try:
            tid = int(payload["task"])
        except (KeyError, TypeError, ValueError):
            raise RepoError("repo_merge needs a numeric 'task'")
        t = self._task_row(tid)
        if t["repo_id"] != r["id"]:
            raise RepoError(f"task {tid} does not belong to {r['name']}")
        if t["status"] != "ready":
            raise RepoError(f"task {tid} is {t['status']}, not ready")
        return tid

    @staticmethod
    def _check_branch(r: dict, branch: str) -> None:
        if not re.fullmatch(r"[\w./-]+", branch) or branch.startswith("-"):
            raise RepoError(f"bad branch name {branch!r}")
        if not subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"], cwd=r["path"],
                              capture_output=True).returncode == 0:
            raise RepoError(f"no local branch {branch}")

    def create_action(self, kind: str, repo: str, payload: dict | None = None, created_by: str = "gemini") -> dict:
        if kind not in REPO_ACTIONS:
            raise RepoError(f"unknown action {kind}")
        r = self.get(repo)
        payload = dict(payload or {})
        if kind == "repo_push":
            payload["branch"] = payload.get("branch") or r["default_branch"]
            self._check_branch(r, payload["branch"])
        if kind == "repo_install" and not PKG_RE.fullmatch(payload.get("package") or ""):
            raise RepoError("bad package spec")
        if kind == "repo_merge":
            self._mergeable(r, payload)
        summary = {"repo_merge": f"Merge task #{payload.get('task')} into {repo}'s {r['default_branch']} (local) and re-run the tests",
                   "repo_undo_merge": f"Undo the last assistant merge in {repo}",
                   "repo_push": f"Push branch {payload.get('branch')} of {repo} to origin",
                   "repo_install": f"pip install {payload.get('package')} into {r['venv_python']}"}[kind]
        aid = existing_action(self.con, kind, repo, payload)
        if aid is None:
            aid = db.insert(self.con, "pending_actions", kind=kind, service=repo, payload=json.dumps(payload),
                            created_by=created_by, status="pending", created_at=db.now())
        return {"needs_confirmation": True, "action_id": aid, "kind": kind, "service": repo, "summary": summary}

    async def confirm_action(self, aid: int) -> dict:
        row = claim_action(self.con, aid)
        p = row["payload"]
        try:
            if row["kind"] == "repo_merge":
                res = await self.merge_task(self._mergeable(self.get(row["service"]), p))
            elif row["kind"] == "repo_undo_merge":
                res = await self.undo_merge(row["service"])
            elif row["kind"] == "repo_push":
                res = await self.push(row["service"], p.get("branch"))
            elif row["kind"] == "repo_install":
                res = await self.install_package(row["service"], p.get("package"))
                t = db.one(self.con, "SELECT id, status FROM repo_tasks WHERE id=?", p.get("task")) if p.get("task") else None
                if t and t["status"] == "needs_package":
                    self._set(t["id"], error="package installed — tap Continue")
            else:
                raise RepoError(f"{row['kind']} is not a repo action")
        except Exception as e:
            finish_action(self.con, aid, error=e)
            raise
        finish_action(self.con, aid, result=res)
        return res
