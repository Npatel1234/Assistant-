# Assistant: a self-hosted Gemini "second brain"

A personal assistant web app for one developer who runs many AI/ML projects at once. You chat with Gemini from your laptop or phone; it remembers your projects, ideas, decisions and reminders over months, delegates code work to Claude Code, and can manage the tmux services, GitHub repos and shell on the machine it runs on.

It is a single-user tool for a trusted home or lab network: FastAPI + SQLite + vanilla JS, no build step, no external database. Read [SECURITY.md](SECURITY.md) before exposing it anywhere.

## Why

Running a dozen experiments means losing track of them: which one stalled, what you decided last Tuesday, which server is up, what Claude Code was in the middle of. This app keeps that state for you and turns it into one chat box. Gemini is the front-of-house assistant (cheap, fast, tool-calling); Claude Code does the heavy code work inside your project folders.

## Features

- **Chat with memory.** Short-term: the last 20 messages plus a rolling summary of older ones. Long-term: durable facts are extracted after each turn, embedded, de-duplicated by similarity and recalled by vector search (keyword fallback when embeddings are unavailable). A profile of you is rewritten nightly.
- **Projects, ideas, decisions, reminders, links.** "idea: ...", "remind me Friday 5pm to ...", "log a decision for project X". Projects have goal, status, next step and notes; a "what should I work on?" ranking and a per-project resume brief are built from your data.
- **Daily and weekly recaps.** Written by a nightly job (weekly on Sunday night); the greeting when you open a chat mentions due reminders, stalled projects, finished repo tasks, downed services and paused Claude sessions.
- **Claude Code history sync.** Every 3 hours the app reads your Claude Code transcripts and updates project records, decisions and memories from them.
- **Claude Code delegation.** Ask a question about the code in a project folder and Gemini runs Claude Code there (full access). Every run is a **live session** you can watch, steer with follow-up messages or stop.
- **Usage-limit pause and resume.** If Claude hits its usage limit the session shows when it resets; tap **Resume at reset** or tell the assistant to resume it, and it continues automatically.
- **tmux service manager.** Register scripts or commands as services; see running state, port, GPU use and last output line; start (dependencies first, GPU-fit check), stop, restart, send input. Describe a config change in words and Claude edits the files after a backup; you review the diff and tap **Apply & restart** or **Undo**.
- **Repo workspace (coder + tester).** Clone a GitHub repo, have Claude build a fast pytest suite, then run each change request on its own branch with tests before and after, regressions highlighted and a behaviour summary. Merge locally, pull upstream with a regression check, push on confirmation.
- **Web terminal.** A real shell in the browser, backed by tmux so it survives page reloads. Needs a second unlock with the app password.
- **Pending-action Confirm model.** Risky operations (stop, restart, send input, register, merge, undo merge, push, package install, HTTP POST calls to services) never run from chat. Gemini can only create a pending action; it runs when you tap **Confirm** in the UI.
- **Prompt-injection guard.** After the model reads untrusted content (web pages, saved links, terminal output, repo text) in a turn, tools that start Claude Code or change things are blocked until you tap **Allow**.
- **Gemini model fallback.** An ordered model list; a model that hits a per-minute or per-day quota is skipped, and the header chip shows which model answered.

Screenshots are intentionally omitted; the UI is a single mobile-friendly page with tabs for Chat, Today, Projects, Ideas, Memory, Claude, Services, Repos and Terminal.

## Requirements

- Python 3.11 or newer and [uv](https://docs.astral.sh/uv/)
- git
- tmux (services and the web terminal) and `ss` from iproute2 (port checks)
- A Gemini API key ([Google AI Studio](https://aistudio.google.com/))
- Optional: the [Claude Code](https://claude.com/claude-code) CLI (`claude`) on `PATH` for delegation, live sessions, config changes and repo tasks
- Optional: `nvidia-smi` for GPU stats and fit checks
- Linux (the app reads `/proc` and uses pty/tmux)

## Quick start

```bash
git clone https://github.com/Npatel1234/Assistant-.git Assistant
cd Assistant
cp .env.example .env
# edit .env: set GEMINI_API_KEY, APP_PASSWORD and SESSION_SECRET
python -c "import secrets; print(secrets.token_urlsafe(32))"   # use this for SESSION_SECRET
uv run python main.py
```

Open `http://<host>:8765` (`http://localhost:8765` on the same machine; `hostname -I` shows the LAN address for your phone). If a phone cannot connect, allow the port in your firewall, for example `sudo ufw allow 8765/tcp`. The app exits at startup if `GEMINI_API_KEY`, `APP_PASSWORD` or `SESSION_SECRET` is empty. It listens on all interfaces (`0.0.0.0`).

## Configuration

Settings come from `.env` (next to `main.py`) and can be overridden by environment variables of the same name. Keys are case-insensitive. `.env.example` lists them all. In `.env`, `#` preceded by a space starts a comment, so values cannot contain ` #`.

| Setting | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | (empty, required) | Gemini API key |
| `APP_PASSWORD` | (empty, required) | Login and terminal-unlock password |
| `SESSION_SECRET` | (empty, required) | Secret that signs the login cookie |
| `MODELS` | `gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-3.5-flash,gemini-2.5-flash` | Chat models, tried in order |
| `EMBED_MODEL` | `gemini-embedding-2` | Embedding model; do not change after first run |
| `SEARCH_MODELS` | `gemini-2.5-flash` | Models for grounded web search |
| `RPM_LIMIT` | `15` | Local calls-per-minute cap per model before falling to the next |
| `PORT` | `8765` | Listen port |
| `TIMEZONE` | `Asia/Kolkata` | Display timezone, reminders, nightly job (data is stored in UTC) |
| `CLAUDE_TIMEOUT` | `3600` | Seconds allowed per Claude turn |
| `CLAUDE_WAIT` | `180` | Seconds the chat waits before "still running" |
| `CLAUDE_IDLE_MIN` | `30` | Idle minutes before a Claude session closes |
| `CLAUDE_MAX_SESSIONS` | `3` | Concurrent Claude turns |
| `REPO_TEST_TIMEOUT` | `300` | Seconds per test run in a cloned repo |
| `TERMINAL_IDLE_MIN` | `30` | Idle minutes before the terminal locks |
| `SYNC_MAX_CALLS` | `30` | Gemini calls per history-sync run |
| `DB_PATH` | `<project>/assistant.db` | SQLite file |
| `CLAUDE_HISTORY_DIR` | `~/.claude/projects` (expanded at import) | Claude Code transcripts to sync; use an absolute path if you override it |

## Using it

- Chat: "idea: ...", "remind me Friday 5pm to ...", "what should I work on?", "in my-project, which providers do I use?" (runs Claude Code in that folder).
- Tell it where a project lives: "my bot is at /path/to/my_bot, main file server.py".
- Claude Code runs with **full access** (`bypassPermissions`) in registered folders. Every run is logged on the project page with changed files.
- Quota: models are tried in order; per-minute limits (and 5xx errors) skip a model for 60 s, per-day limits until midnight Pacific. If every model is exhausted you get the time it comes back.

### Live Claude sessions

- Every Claude Code run starts a live session. In chat, tap **watch live** on the tool line, or open the Claude tab.
- The transcript streams Claude's messages and tool calls; tap a tool line for its output. Type below to send a follow-up in the same conversation; **Stop** kills the process and everything it started.
- Up to `CLAUDE_MAX_SESSIONS` turns run at once (one session per project); others show **queued**. Idle sessions close after `CLAUDE_IDLE_MIN` minutes and resume later with the same conversation.
- Usage limit: the session shows **paused** with the reset time (your timezone) and the top of the tab shows usage. Tap **Resume at reset** or tell the assistant "resume it when usage resets".

### Services (tmux)

- The Services tab lists your registered services and every other tmux session (view only): running?, port up, GPU use, last output line, plus a GPU bar.
- **Import** finds `run_*.sh` scripts in your registered project folders; **+ Service** registers a script or a folder plus command.
- Tap a service for its live terminal (refreshes every 2 s), a command box and Ctrl-C, Start / Restart / Stop, **Summarize** (Gemini) and **Investigate** (Claude Code).
- **Change config**: describe the change ("use a different LLM provider"); Claude edits the service's files after they are backed up; you see the diff and tap **Apply & restart** or **Undo** (restores the files byte-for-byte).
- In chat, the assistant can start services (dependencies first, GPU-fit check), but stop / restart / typing / registering show **Confirm** buttons; nothing happens until you tap.

### Repos (GitHub coder + tester)

- **Repos > + Repo** (or tell the assistant a GitHub URL): clones into `workspace/<name>`; give it an existing venv that can run the app (pytest is added from `workspace/.testtools`; your venv is not modified).
- **Build test suite**: Claude writes `tests_assistant/` (fast, faked models/network) on branch `assistant/tests`; review it and tap **Accept suite**.
- **Start task**: each request gets a branch, tests before, Claude's change (watch live), tests after, regressions in red and a behaviour summary. **Merge to main** (local) re-runs the tests; **Undo merge** resets main if nothing moved since.
- **Pull** merges upstream into the local main and reports regressions; **Push** never force-pushes and uses your git credentials. From chat, merge / undo / push / package installs show **Confirm** buttons.
- If a post-merge or post-pull check fails, times out or has no baseline, the UI shows a warning with the reason instead of implying "no regressions".
- **Continue**: a task paused by the Claude usage limit, stopped half-way, or waiting for a package (`NEEDS_PACKAGE`) gets a Continue button; package installs (`pip install` into your venv) need a Confirm tap first. Half-done work is kept as a WIP commit on the task branch, never on main.
- One operation per repo at a time: anything else waits up to 10 s, then says the repo is busy. **Accept suite** is refused if the suite branch changes files outside `tests_assistant/`.
- In chat, anything read from a repo marks that turn untrusted, so "clone X" and "build its tests" take two messages (or tap Allow).
- `workspace/pytest.ini` is created on purpose: it keeps this app's pytest settings out of your repos' test runs.

### Terminal

- The Terminal tab is a real shell on this server (your user, home folder, login shell). Unlock with the app password; it locks again after `TERMINAL_IDLE_MIN` minutes without typing, on **Lock**, on logout or on restart.
- **+ New** opens more terminals (tmux sessions `webterm-1`, `webterm-2`, ...); tap a tab to switch. They keep running when you switch, reload or close the page; the close button ends one. A key bar has Esc/Tab/Ctrl/arrows for phones.
- Only you can use it: Gemini can't read or type into it, and the Services tab hides `webterm-*` sessions.

## Run on boot (systemd user service)

Use your own paths for `/path/to/Assistant` and `~/.local/bin/uv` (`which uv`).

```ini
# ~/.config/systemd/user/assistant.service
[Unit]
Description=Gemini personal assistant
After=network-online.target

[Service]
WorkingDirectory=/path/to/Assistant
ExecStart=%h/.local/bin/uv run main.py
Environment=PATH=%h/.local/bin:/usr/local/bin:/usr/bin:/bin
Restart=on-failure

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now assistant
loginctl enable-linger $USER   # keep running when logged out
journalctl --user -u assistant -f
```

Make sure `claude`, `tmux` and `git` are on the `PATH` set above if you use them.

## Project layout

```
main.py            FastAPI app, auth, JSON API, SSE chat, WebSocket terminal, starts the scheduler
agent.py           Chat turn loop (Gemini function calling), system prompt, taint guard, greeting
tools.py           Gemini-callable tool registry
memory.py          Long-term memory: dedup writes, vector recall, related items
llm.py             Gemini wrapper: model fallback, RPM throttle, quota state, embeddings
claude_sessions.py Live Claude Code sessions (stream-json), usage-limit pause/resume
services.py        tmux service manager, config changes with backup/diff/undo, pending actions
workspace.py       Repo workspace: clone, test suite, tasks, merge/pull/push
terminal.py        tmux-backed web terminal over a pty
jobs.py            History sync, nightly reflection/recaps/backup, scheduler tick
db.py              SQLite schema and helpers
config.py          Settings from .env + defaults
static/            index.html (single-page UI), login.html
test_core.py       Test suite
docs/              Architecture notes and design specs/plans
```

Runtime data (git-ignored): `assistant.db` (+ WAL files), `assistant.log`, `backups/` (nightly DB copies, last 7; config-change backups in `backups/service_changes/`), `workspace/` (cloned repos).

## Testing

```bash
uv run pytest -q test_core.py -W error
```

The tests use fakes for Gemini, Claude, tmux and git.

## Security

Single user, trusted network, plain HTTP, and the app can run full-access Claude Code and a shell. Choose a strong `APP_PASSWORD`, keep it off the public internet, and read [SECURITY.md](SECURITY.md) for the threat model, the confirmation model and how to report issues.

## Docs

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): modules, flows, data model
- [docs/superpowers/specs/](docs/superpowers/specs/): design specs (assistant, live Claude sessions, tmux service manager, repo workspace)
- [docs/superpowers/plans/](docs/superpowers/plans/): implementation plans
