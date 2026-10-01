# Repo Workspace (GitHub coder + tester) — Design Spec

Date: 2026-09-30
Status: Draft for review
Builds on: master (live Claude sessions `claude_sessions.py`, service manager `services.py`, pending actions)
Part 3 of 3.

## 1. Goal

The user gives a GitHub URL; the assistant clones it into `ai-new-projects/workspace/`, has Claude
build a fast pytest suite for it, then carries out change requests as branch-per-task work: tests
before, Claude's change (live, watchable), tests after, regressions highlighted, a plain-language
behaviour summary. Merging to the local `main` (and pulling from GitHub) is followed by a
regression check with undo. Pushing happens only when the user asks and taps Confirm.

### Success criteria
- `add_repo(url, venv)` clones + registers; import check passes in the given venv.
- `build_test_suite` produces `tests_assistant/` that passes on `main`, reviewed and accepted by the user.
- A task on `RAG_for_192_server` (sandbox repo the user created for this) produces a branch, a
  before/after test report with regressions (if any), and a summary; Merge + check works; Push
  to origin works after Confirm.

## 2. Verified environment facts (2026-09-30)

- `gh` CLI not installed. `git` credential helper = `cache` (token cached in memory; may expire →
  push then fails with git's error, which is shown). `git push --dry-run origin <branch>` to the
  sandbox repo succeeded.
- Sandbox clone: `workspace/RAG_for_192_server` (branches `main`, `assistant-test`), 119 MB, no
  pytest suite (only live-server scripts like `Test_RAG.py`).
- User's venv `/mnt/newdisk/neel/Backend_rag_server/ai-rag/venv` (Python 3.12.3): has fastapi,
  httpx 0.28.1; **no pytest**. `import main` from the clone works in 6.4 s (FastAPI app, no model
  load at import).
- Test tools overlay works: `uv pip install --python <venv python> --target <dir> pytest`, then
  `PYTHONPATH=<dir> <venv python> -m pytest --junitxml=...` — the user's venv is not modified.
- Docker is available (not used in v1).

## 3. Scope

In: repo registry + clone, test-tools overlay, Claude-built suite (accept flow), test runs with
per-test results, tasks on branches with before/after/regressions/summary, fix-regressions,
discard, local merge + check + undo, pull + check, push (confirm), known failures, Repos tab,
chat cards, greeting mention of upstream regressions.
Out: GitHub PRs / gh CLI, parallel tasks per repo (worktrees), Docker isolation, CI integration,
non-Python repos (runner is pytest-only in v1).

## 4. Architecture

New module `workspace.py` (`RepoManager`), using asyncio subprocesses for git/pytest (same env
handling as services), `SessionManager` for Claude work, `ServiceManager.create_action`-style
pending actions (shared `pending_actions` table with new kinds).

- Paths: every repo path must resolve inside `config.ROOT / "workspace"` (`WORKSPACE_DIR`).
- Test tools overlay: `workspace/.testtools/py<major.minor>/` built once per Python version via
  `uv pip install --python <venv python> --target <dir> pytest` (+ `pytest-timeout`); runs use
  `PYTHONPATH=<overlay>:<repo>` and `ASSISTANT_TEST=1`.
- Test run: `<venv python> -m pytest tests_assistant -q -p no:cacheprovider
  --junitxml=<tmp>/report.xml -o junit_family=xunit2` in the repo, own process group, timeout
  `REPO_TEST_TIMEOUT` (300 s) → group killed. Parse JUnit XML → `{nodeid: passed|failed|error|skipped}`;
  fallback: pytest summary line.
- Claude work runs via `SessionManager.run(project_for_repo, prompt, follow_up)` with the repo as
  cwd (repos are registered as projects).

## 5. Data model

- `repos(id, name UNIQUE, url, path UNIQUE, venv_python, default_branch, suite_status
  [none|building|proposed|accepted|failed], suite_branch, known_failures TEXT (JSON list),
  env_status, env_error, project_id, created_at, updated_at)`
- `repo_tasks(id, repo_id, request, branch, status [queued|testing_before|claude|testing_after|
  ready|failed|merged|discarded|needs_package|paused_limit], base_sha, head_sha, before_run,
  after_run, regressions TEXT, new_tests TEXT, fixed_tests TEXT, summary, diff_stat, claude_session,
  error, merge_prev_sha, merge_sha, created_at, updated_at)`
- `test_runs(id, repo_id, task_id, ref, sha, kind [baseline|before|after|merge|pull|manual],
  passed, failed, errors, skipped, results TEXT (JSON nodeid→outcome), duration_s, output_tail,
  created_at)`
- `pending_actions` kinds added: `repo_merge`, `repo_undo_merge`, `repo_push`, `repo_install`
  (payload holds repo/task/branch/package).

## 6. Flows

### 6.1 Add repo
`add_repo(url, venv=None, name=None)`: name from URL; if `workspace/<name>` exists and its
`origin` URL matches, reuse; else `git clone <url>`. Detect default branch
(`git symbolic-ref refs/remotes/origin/HEAD`). `venv` = path to a venv dir or python; stored as
`venv_python`. Env check: `<python> -c "import sys; sys.path.insert(0, '.'); import <entry>"` where
`<entry>` defaults to `main` (Claude may adjust during suite build) — failure → `env_status=error`
with the message. Register a project (path = clone) for Claude sessions.

### 6.2 Build the test suite
Requires a clean worktree on `main`. Create/reset branch `assistant/tests` from `main`. Claude
prompt: write `tests_assistant/` (conftest + tests) using FastAPI TestClient for each endpoint,
faking LLM/embedding/vector store/network (no real model loads, no GPU, no outbound calls; must
honour `ASSISTANT_TEST=1`), deterministic, < 60 s total; run it with the given command (provided
in the prompt) and iterate until green; do not `pip install`; commit on `assistant/tests`.
Then the assistant runs the suite itself (baseline). All green → `suite_status=proposed`. Still
failing after Claude's turn → second Claude turn with the failures; after 3 turns remaining
failures become `known_failures` and status `proposed` (flagged in the card). User taps
**Accept suite** → merge `assistant/tests` into `main` (`--no-ff`, local) → `accepted`.

### 6.3 Task
Preconditions: suite accepted, worktree clean, no other active task on the repo.
1. Branch `task/<id>-<slug>` from `main`; `base_sha`.
2. Before run on the branch (identical to main) → `before_run`.
3. Claude (fresh conversation): make the change; add/update tests in `tests_assistant/` for the
   new behaviour; run the suite; don't `pip install` (if a package is needed, say
   `NEEDS_PACKAGE: <name>` and stop); commit on the branch.
4. After run → `after_run`; head_sha; `diff_stat` (`git diff --stat main...branch`).
5. Regressions = tests `passed` before and `failed|error` after, minus `known_failures`;
   new tests = present only after; fixed = failed before, passed after.
6. Summary: Claude (same session, follow-up) writes 3–6 bullets on behaviour changes.
7. Status `ready` (or `needs_package` if Claude reported one → pending `repo_install` action).
Long-running: the tool starts a background task and returns `{task_id, status}` immediately;
the Repos tab polls; the greeting reports finished tasks since last visit.

### 6.4 Fix regressions
Resume the task's Claude session with the failing tests + their output tails; after run again;
recompute.

### 6.5 Merge + check (Confirm)
Clean worktree required. `merge_prev_sha = main`; `git merge --no-ff task/...` on `main`
(conflict → `git merge --abort`, task error "conflicts with main"). Run suite (kind `merge`);
compare with the task's after run and main's previous run → regressions reported. Status `merged`.
**Undo merge** (Confirm): only if `main` HEAD == `merge_sha` → `git reset --hard merge_prev_sha`
(local only); else refuse ("main moved since the merge").

### 6.6 Pull + check
`git pull --ff-only` on `main` (non-ff → error, no merge commits created by pulls). If HEAD moved:
run suite (kind `pull`), compare with the previous main run → regressions; Claude summary of the
upstream commits' behaviour changes (`git log -p old..new`, capped). Greeting mentions upstream
regressions.

### 6.7 Push (Confirm)
`git push origin <branch>` (never `--force`/`--force-with-lease`), `GIT_TERMINAL_PROMPT=0`,
60 s timeout; show git's output on failure (e.g. expired cached credentials).

### 6.8 Discard
Delete `task/*` branch (`git branch -D`) only for tasks not merged; status `discarded`.

## 7. Safety

- Paths confined to `WORKSPACE_DIR`; branch names generated (`task/<id>-<slug>` with slug
  `[a-z0-9-]{1,40}`), never user-supplied raw.
- No `--force` anywhere; `reset --hard` only in undo-merge with the SHA guard.
- Worktree must be clean before task/merge/pull/suite (no silent stash).
- User's venv is never modified by the assistant except through a confirmed `repo_install`
  (`<venv python> -m pip install <package>`), package name validated (`^[A-Za-z0-9._-]+(==[\w.]+)?$`).
  `pip freeze` recorded before/after each task; a difference is shown as a warning.
- Tests run with `ASSISTANT_TEST=1`, own process group, timeout-kill of the whole group.
- Guarded (injection) tools: `start_repo_task`, `build_test_suite`, `fix_task_regressions`,
  `add_repo`; `repo_merge`, `repo_undo_merge`, `repo_push`, `repo_install` are pending actions.

## 8. Gemini tools

`add_repo(url, venv=None, name=None)`, `build_test_suite(repo, focus=None)`,
`run_repo_tests(repo, ref=None)`, `start_repo_task(repo, request)`,
`fix_task_regressions(task_id)`, `list_repo_tasks(repo=None)`, `repo_task(task_id)`,
`merge_task(task_id)` → pending, `undo_merge(repo)` → pending, `pull_repo(repo)`,
`push_repo(repo, branch=None)` → pending, `discard_task(task_id)`.

## 9. API

`GET/POST /api/repos`, `GET /api/repos/{name}`, `POST /api/repos/{name}/suite`,
`POST /api/repos/{name}/suite/accept`, `POST /api/repos/{name}/test {ref?}`,
`POST /api/repos/{name}/tasks {request}`, `GET /api/repos/{name}/tasks`,
`GET /api/tasks/{id}` (incl. diff capped 60k), `POST /api/tasks/{id}/fix`,
`POST /api/tasks/{id}/merge`, `POST /api/tasks/{id}/discard`, `POST /api/repos/{name}/undo-merge`,
`POST /api/repos/{name}/pull`, `POST /api/repos/{name}/push {branch?}`,
`GET /api/repos/{name}/runs`. UI routes act directly (the user tapped); Gemini goes through
pending actions.

## 10. UI — 🐙 Repos tab

Repo cards (branch, last main run ✓/✗ + age, open tasks, suite status, env status). Detail sheet:
Run tests / Pull / Push main / New task; suite card (Build / diff / Accept, known failures);
runs history; task cards (status chip, before→after counts, regressions in red, new/fixed tests,
summary (md), collapsible diff, Watch live, Merge, Fix regressions, Discard). Chat renders task
cards and pending-action Confirm cards.

## 11. Error handling

Clone failure → git stderr. Env check failure → repo registered with `env_status=error`.
Unclean worktree → refuse with `git status --short`. Merge conflict → abort + card. Pull non-ff
→ error. Test timeout → run `error` with tail; JUnit missing → summary-line fallback.
Claude limit → task `paused_limit` (resume via existing flow, task continues when the session
resumes — v1: user taps "Continue task" after the reset).

## 12. Testing

Real git on throwaway repos (tmp dir): a bare "origin" + a tiny FastAPI app with its own passing
test file used as the suite; overlay replaced by the test interpreter already having pytest.
Cases: add repo (clone, reuse, bad URL); path confinement; run tests → per-test results;
task with fake Claude that breaks a test → regression listed; fix → cleared; known failures
excluded; merge + check; merge that regresses + undo; undo refused when main moved; merge
conflict aborted; pull with upstream breaking commit → regression; push to bare origin after
confirm only; discard; unclean worktree refused; test timeout kills group; needs_package →
pending install action; pip freeze diff warning.
Live (controller): `RAG_for_192_server` + ai-rag venv: real suite build, one real task, merge,
push to the sandbox origin.
