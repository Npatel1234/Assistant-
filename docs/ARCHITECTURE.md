# Architecture

One Python process (FastAPI under uvicorn) serves a static single-page UI, a JSON API, an SSE chat stream and a terminal WebSocket. State lives in one SQLite file (WAL mode). Background work runs as asyncio tasks inside the same process. External processes are `claude` (Claude Code), `tmux`, `git`, pytest in a user venv, and optionally `nvidia-smi`.

```
 browser (static/index.html)
    | fetch + SSE + polling + WebSocket
 main.py (FastAPI) --- agent.py --- llm.py --> Gemini API
    |                     |
    |                  tools.py --- memory.py
    |                     |
    |        +------------+-------------+--------------+
    |   claude_sessions.py   services.py   workspace.py   (jobs.py: scheduler)
    |        |  claude CLI       | tmux, ports, GPU   | git, pytest, claude
    |   terminal.py (tmux webterm-N over a pty)
    +--- db.py / SQLite (assistant.db)
```

## Modules

| Module | Responsibility |
|---|---|
| `config.py` | Reads `.env` (keys upper-cased, real env vars override) over `DEFAULTS`; exposes typed constants. Timestamps are stored UTC, shown in `TIMEZONE`. |
| `db.py` | The `SCHEMA` (created with `IF NOT EXISTS` on connect), WAL + foreign keys, tiny helpers (`rows`, `one`, `insert`, `update`, `kv_get/kv_set`). |
| `llm.py` | Gemini wrapper: ordered model fallback, local RPM throttle, quota state in `model_state`, `generate`, grounded `search`, `embed`, vector helpers (`pack`, `unpack`, `top_k`). |
| `memory.py` | Long-term memory: `remember` merges near-duplicates (cosine >= 0.9), `recall` returns memories above 0.55 similarity with a keyword fallback, related-item suggestions (0.75). |
| `tools.py` | The tool registry exposed to Gemini (decorator `@tool`), plus ranking and resume-brief helpers. Each tool returns a dict; failures become `{"error": ...}`. |
| `agent.py` | The chat turn loop, system prompt, injection guard, history building, post-turn memory extraction, rolling summary, greeting. |
| `claude_sessions.py` | `SessionManager`: one streaming `claude` process per project, event log, usage-limit pause and resume. |
| `services.py` | `ServiceManager`: tmux service registry, status, start/stop with dependencies and GPU fit, config changes with backup/diff/undo, the pending-action machinery. |
| `workspace.py` | `RepoManager`: clone repos, Claude-built test suite, branch-per-task before/after test comparison, merge/pull/push with checks. |
| `terminal.py` | `TerminalManager`: tmux sessions `webterm-N` attached through a pty. |
| `jobs.py` | History sync, nightly reflection/recaps/backup, scheduler tick. |
| `main.py` | App factory `create_app`, auth middleware, routes, `main()` wiring. |
| `static/` | `index.html` (whole UI, vanilla JS) and `login.html`. |

## Chat flow

1. `POST /api/chat {conversation_id, text}` starts `agent.start_turn`, which runs the turn as a detached asyncio task feeding an `asyncio.Queue`. The HTTP response is an SSE stream that drains the queue, so closing the tab does not cancel the turn; the saved result appears when the conversation is reopened.
2. `chat_turn` stores the user message, then `build_system` fills the system prompt: timezone and time, the learned profile (`kv`), up to 25 non-done projects, reminders due within a day, memories recalled for this message (`memory.recall`) and the rolling conversation summary.
3. History is the last 20 messages. For assistant messages a short **tool line** is appended (`[tools this turn: name(args) -> ok / error / pending action #N]`, at most 800 characters, never tool output text) so the model knows what it did last turn without replaying results.
4. The loop calls `llm.generate` with the tool declarations and automatic function calling disabled. Up to 6 rounds: if the response has function calls, each is executed through `tools.run_tool`, its result is appended as a function response (the model's own content is kept so Gemini 3 thought signatures survive), and the loop continues; if there are none, the text is the answer. Hitting the cap yields a "ask me to continue" message.
5. SSE events: `tool` (name and args), `tool_result` (result, shortened to about 4 KB with key fields kept), `answer` (text, model, recalled memories, tool log) or `error`. The assistant message is stored with `meta` = model, memories and tool log.
6. If Gemini fails after tools already ran, the tool results are saved as a partial assistant message so the record of what happened is kept. If every model is out of quota the error says when quota returns.
7. After the answer, `after_turn` runs in the background: it extracts up to 5 durable memories (skipping what tools already saved), writes them through `memory.remember`, and refreshes the rolling summary once enough old messages have scrolled out of the 20-message window. Background calls use only the first model so they do not eat chat quota.

### Prompt-injection guard

`chat_turn` keeps a per-turn `tainted` flag. It becomes true after any repo tool, `capture_link`, `web_search`, `service_output`, `investigate_service`, `change_service_config` or `call_service`. Once tainted, tools in `GUARDED_TOOLS` (`ask_claude_code`, `schedule_claude_resume`, `change_service_config`, `investigate_service`, `add_repo`, `build_test_suite`, `start_repo_task`, `fix_task_regressions`, `pull_repo`, `discard_task`) return `{"blocked": true, ...}` instead of running, and `start_service` is turned into a pending Confirm action. The UI shows an **Allow** button on blocked calls; tapping it sends a new user message that authorizes that specific call. Stop, restart, send, register, merge, undo-merge, push and install are not in the guarded set because they only ever create pending actions.

## Tool registry

`tools.py` registers functions with `@tool(description, params, required)`; the registry `TOOLS` yields Gemini `FunctionDeclaration`s via `declarations()` and `run_tool` dispatches by name, converting `ToolError`/`ServiceError` into error dicts. Groups:

- Memory and tracking: `save_memory`, `search_memory`, `create_project`, `update_project`, `list_projects`, `save_idea`, `list_ideas`, `promote_idea`, `set_reminder`, `list_reminders`, `complete_reminder`, `log_decision`, `list_decisions`, `capture_link`, `get_recap`, `what_to_work_on`, `resume_context`, `web_search`.
- Claude: `ask_claude_code`, `list_claude_sessions`, `schedule_claude_resume`.
- Services: `list_services`, `service_output`, `start_service`, `stop_service`, `restart_service`, `send_to_service`, `call_service` (GET runs directly; other methods create a pending action), `register_service`, `change_service_config`, `undo_service_change`, `investigate_service`.
- Repos (`REPO_TOOLS`): `add_repo`, `build_test_suite`, `run_repo_tests`, `start_repo_task`, `fix_task_regressions`, `list_repo_tasks`, `repo_task`, `merge_task`, `undo_merge`, `pull_repo`, `push_repo`, `discard_task`.

## Background jobs (`jobs.py`)

`scheduler` runs `tick` every 60 seconds inside the FastAPI lifespan. Each tick:

1. If a session manager exists: check the clock against Google's `Date` header (hourly, stored in `kv`; warns above 120 s drift) and fire any due scheduled Claude resumes (first, so they never wait behind a long sync).
2. `services.health_tick()`: records up/down transitions for services that should be running.
3. **Claude history sync** when `kv.next_sync` is due (every 3 h; retried after 10 min if deferred or busy): parses transcripts under `CLAUDE_HISTORY_DIR`, skipping unchanged files (hash in `sync_state`) and scratch directories (temp dir, `$HOME`, the app's own `workspace/`), asks Gemini to update the project record and extract decisions and memories, capped at `SYNC_MAX_CALLS` per run.
4. **Nightly** after 02:00 local time, once per local day: rewrite the profile, write the daily recap (and weekly on Sunday night), backfill missing embeddings, merge duplicate memories, mark stalled projects, prune old Claude event payloads (older than 14 days shrunk to previews) and back up the DB to `backups/` (last 7 kept). Each step is isolated so one failure does not stop the others.

## Data model (`db.py` SCHEMA)

| Table | Purpose |
|---|---|
| `conversations` | Chat threads with title, rolling `summary` and `summary_upto` marker. |
| `messages` | Messages per conversation; `meta` JSON holds model, recalled memories and tool log. |
| `projects` | Tracked projects: path, main files, goal, status, next step, notes, activity time, embedding. |
| `memories` | Long-term facts/preferences/decisions/notes with embeddings and optional project link. |
| `ideas` | Captured ideas with tags, status (inbox/promoted/dropped) and embedding. |
| `reminders` | Reminders with due time (UTC) and done flag. |
| `decisions` | Per-project decisions with the reason and source. |
| `links` | Captured URLs with title and summary. |
| `recaps` | Generated daily and weekly recaps. |
| `claude_runs` | Log of Claude Code runs: prompt, output, files changed, exit code, duration. |
| `sync_state` | File hash per Claude transcript already synced. |
| `model_state` | Per-model "exhausted until" timestamps for quota fallback. |
| `kv` | Small key/value store (profile, last seen, next sync, clock offset, service up/down flags). |
| `claude_sessions` | Live Claude sessions per project: status, pending message, resume time and flags. |
| `claude_events` | Ordered event log (`seq`) of each session's stream, for replay in the UI. |
| `services` | Registered tmux services: script/command, cwd, venv, port, dependencies, GPU memory, `should_run`. |
| `service_events` | Audit trail of service actions (start, stop, send, ...). |
| `service_changes` | Claude-made config changes: request, status, backup dir, manifest, diff, explanation. |
| `pending_actions` | Confirm-model actions: kind, target, payload, status, result. |
| `repos` | Cloned repos: URL, path, venv python, default branch, suite status, known failures, last pull. |
| `repo_tasks` | Branch-per-task work: status, shas, before/after runs, regressions, summary, merge info. |
| `test_runs` | Per-run test results (counts, per-test outcomes, output tail) for repos. |

## Claude sessions (`claude_sessions.py`)

- Each project gets at most one session. A turn spawns `claude -p --input-format stream-json --output-format stream-json --verbose --permission-mode bypassPermissions` in the project folder, in its own process group (`start_new_session`). With stream-json in and out the process stays alive across turns: each stdin line is a user turn and each turn ends with a `result` event.
- Every stream event is appended to `claude_events` (payloads over 64 KB, and later ones past a soft cap of 5000 events, are truncated). The UI replays and follows them via SSE.
- Status values: `queued`, `starting`, `running`, `idle`, `paused_limit`, `closed` (idle timeout; resumes later with the same Claude conversation), `stopped`. At most `CLAUDE_MAX_SESSIONS` turns run concurrently; each turn is bounded by `CLAUDE_TIMEOUT`.
- Stop and timeouts kill the whole process group, including anything Claude started.
- **Usage limit:** each turn reports a `rate_limit_event` with status and reset time; a limit message is also parsed (`|<epoch>` or "resets 7pm (TZ)"). On a limit the session becomes `paused_limit`, keeps the unsent message as `pending_message`, and stores `resume_at` (estimated as +5 h if the reset is unknown). **Resume at reset** (or `schedule_claude_resume`) sets `resume_scheduled`; the scheduler tick fires due resumes by sending the pending message.

## Services (`services.py`)

- Services are tmux sessions the app starts by typing the command (after sourcing the venv's `activate` if set) into a named session. Status combines `tmux list-sessions`, listening ports (`ss -ltnH`, `port_up`), the last output line and GPU numbers from `nvidia-smi` when available. Unregistered tmux sessions are listed view-only; `webterm-*` sessions are hidden.
- `start` brings up dependencies first, waits for the port, and checks GPU fit against remembered per-service memory (learned from `nvidia-smi` per-process usage once the service runs). A GPU shortfall returns an error the user can override.
- **Config changes:** `change_config` backs up the service's tracked files (size and extension limits, including `.env` files) into `backups/service_changes/`, runs Claude Code on the service folder with the request, then compares file hashes to build a manifest and unified diffs, flags edits outside the tracked set (a flagged change cannot be applied, only undone) and stores an explanation. Status `proposed`. `apply` marks `applying` then `applied` and restarts the service; `undo` restores the backed-up files byte-for-byte (`undone`, or `undone_partial` if files changed since and conflicts were skipped).
- Gemini cannot type into panes freely: stop, restart, send, register, start-with-GPU-override and non-GET HTTP calls become pending actions.

## Repo workspace (`workspace.py`)

- Repos are cloned into `workspace/<name>` (paths must resolve inside it). A pytest overlay `workspace/.testtools/py<ver>` is installed with `uv pip install --target` so the user's venv is not modified; runs use that `PYTHONPATH`.
- **Suite:** `build_suite` has Claude write `tests_assistant/` on branch `assistant/tests`; `accept_suite` verifies the branch only touched `tests_assistant/`, merges it into the default branch and runs a baseline suite.
- **Task:** `start_task` creates branch `task/<id>-<slug>`, runs the suite before (baseline per sha), runs Claude (live session, up to a few attempts), runs the suite after, then `compare_runs(before, after, known)` computes `regressions` (passed before and failed, errored or vanished now, minus known failures), `new_tests`, `fixed_tests` and `new_failing`. A summary of behaviour changes is generated by Claude. Statuses include `queued`, `testing_before`, `claude`, `testing_after`, `ready`, `failed`, `paused_limit`, `needs_package`. `fix_task` continues or repairs regressions.
- **Merge / pull / push:** `merge_task` merges locally into the default branch and re-runs tests (with undo if nothing moved); `pull` merges upstream and reports regressions; `push` never force-pushes. Merge, undo, push and package install are pending actions.
- A per-repo lock gives one operation at a time (waits up to 10 s, then reports busy). Test runs execute in their own process group with `REPO_TEST_TIMEOUT`, killed as a group on timeout. Tasks stuck in an active status at startup are marked failed ("interrupted by restart").

## Web terminal (`terminal.py` and the WebSocket route)

Each terminal is a tmux session `webterm-N` running the login shell. `attach` starts a tmux client on a pty; the WebSocket relays output to the browser and keystrokes (`{"t":"in","d":...}`) and resizes (`{"t":"resize",...}`) to the pty. Closing the browser kills only the client (its process group), never the session. Access needs the login cookie plus an in-memory unlock (password re-entry), with idle expiry; the WebSocket handler repeats all checks before `accept()`, including an `Origin` equal to `Host`. Lock, logout, or restart revokes the unlock and the open socket closes within about 2 seconds.

## Pending actions

`pending_actions` rows are created by `ServiceManager.create_action` and `RepoManager.create_action` (kinds: `stop`, `restart`, `send`, `register`, `start`, `start_override_gpu`, `http`, `repo_merge`, `repo_undo_merge`, `repo_push`, `repo_install`). Lifecycle: `pending` -> (`POST /api/actions/{id}/confirm`) -> `running` -> `done` or `failed`; or `cancelled` via `/cancel`; or `expired` when older than 15 minutes at confirm time. `claim_action` atomically checks validity and marks `running` so a double tap cannot run twice. Only the UI (authenticated HTTP) can confirm; there is no tool that confirms. The UI then sends the model an `[Action #N ...]` message reporting the outcome so it can continue the user's request. Running actions interrupted by a restart are marked failed.

## UI and API

The UI is one page (`static/index.html`) with tabs Chat, Projects, Claude, Services, Repos, Terminal, Ideas, Memory, Today. It authenticates by a signed cookie set from `static/login.html`.

- **Chat:** `fetch` POST to `/api/chat` and read the SSE body with a stream reader.
- **Live Claude transcripts:** `EventSource` on `/api/claude/sessions/{id}/stream?after=<seq>`.
- **Polling:** service list every 10 s, service pane output every 2 s, repo view every 3 s, reminder badge every 60 s.
- **Terminal:** WebSocket `/api/terminals/{name}/ws`.
- Everything else is plain JSON `fetch` against `/api/...` routes in `main.py` (conversations, projects, ideas, memories, profile, today, status, sync, Claude sessions, services, changes, actions, repos, tasks, terminal).

## Auth

Login posts the password to `/login`; success sets an `itsdangerous`-signed cookie (30-day max age, `HttpOnly`, `SameSite=Lax`, `Secure` only over HTTPS). A middleware requires it for everything except `/login`; API calls get 401, pages redirect. Password checks use constant-time comparison and a per-IP limit of 5 failures per 15 minutes (shared by login and terminal unlock).
