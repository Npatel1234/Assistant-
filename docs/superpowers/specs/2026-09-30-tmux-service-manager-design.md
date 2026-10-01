# tmux Service Manager — Design Spec

Date: 2026-09-30
Status: Draft for review
Builds on: the assistant on master (incl. live Claude sessions, `claude_sessions.py`)
Part 2 of 3 (next: #3 GitHub workspace coder + tester)

## 1. Goal

The assistant knows the user's services (each a tmux session started by a `run_*.sh`-style
script), shows what is running (tmux alive, port listening, GPU use, last output), starts/stops/
restarts them (dependencies first, GPU-fit check), shows each pane live in the UI with a command
box, summarizes or investigates output (Gemini / Claude Code), and applies config changes the
user asks for ("run RAG with Gemini") by having Claude Code edit the service's config — with a
diff, backups, confirm-before-restart and one-tap undo.

### Success criteria
- One-time import registers the user's scripts (`run_main.sh`, `run_qwen4b.sh`, `run_app3.sh`,
  `run_jan.sh`, `run_server2.sh`, key-manager) with correct session/cwd/venv/command/port.
- Services tab shows all tmux sessions (managed + unmanaged view-only) with live status and GPU.
- "Start RAG on vLLM" starts `qwen4b` first if needed, waits for port 8008, then starts `main`.
- "Run RAG with Gemini" produces a Claude-made config diff; Apply restarts the service;
  Undo restores files byte-for-byte.
- Stop/restart/apply/Gemini keystrokes never happen without a UI tap by the user.

## 2. Verified environment facts (2026-09-30)

- tmux sessions (real server): app3, claude_coder, claude_timer, keymanager, koko_tts, main,
  qwen4b, qwen_asr, server2. `claude_coder`/`claude_timer` are the user's own → unmanaged.
- Scripts follow: `SESSION="x"`; `tmux has-session`; `tmux new-session -d -s $SESSION`;
  `tmux send-keys -t $SESSION "<cmd>" C-m` lines (cd / `. venv/bin/activate` / command);
  trailing `tmux attach -t $SESSION`.
- Running such a script without a TTY: session is created and the command runs; the script
  exits 1 only because `tmux attach` fails ("open terminal failed: not a terminal"). Success
  must be judged by `tmux has-session`, not the exit code.
- `tmux capture-pane -p -t S -S -N` returns scrollback; `send-keys -t S C-c` interrupts;
  `send-keys -t S -l "<text>"` + `Enter` sends literal text.
- Setting `TMUX_TMPDIR` isolates a tmux server (also for scripts calling plain `tmux`). The
  socket path must be < 108 chars → tests use `mktemp -d /tmp/tx.XXXX`.
- Ports: `ss -ltnp` works without root (8000, 8008, 8764, 8987, 9000 in use).
- GPU: NVIDIA L40S 46 GB; `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`
  gives per-pid MB (37.6 GB used; qwen4b ≈17 GB).
- RAG app: provider via env `DEFAULT_PROVIDER` (`gemini`|`vllm`) + `DEFAULT_COMPANY_ID`, or
  `config_V2.json` `default_provider`, or live `POST /switch_llm {"provider": …}` on port 8764.

## 3. Scope

In: registry + import, status, live pane view + command box + Ctrl-C, start/stop/restart with
deps and GPU fit, summarize (Gemini) / investigate (Claude), Claude-driven config changes with
backup/diff/apply/undo, pending-action confirmations, health check + greeting mentions.
Out: non-tmux services (systemd/docker), remote hosts, automatic restarts on crash (only
reported), editing unmanaged sessions.

## 4. Architecture

New module `services.py` (tmux/ss/nvidia-smi via asyncio subprocesses; all calls pass one env
so tests can set `TMUX_TMPDIR`):

- `Registry` over table `services`.
- `parse_script(path) -> dict` — regex for the pattern in §2; fallback: ask Claude Code (via
  `SessionManager.run`) to return JSON fields; user confirms before saving.
- `status() -> list[dict]` — all tmux sessions (`list-panes -a -F`), managed + unmanaged; for
  each: alive, pane command, pane pid, port listening, GPU MB of the pane's process tree,
  last non-empty output line, `should_run` flag.
- `output(name, lines=300)`, `send(name, text=None, ctrl_c=False)` (managed only).
- `start(name, override_gpu=False)` — recursively start `depends_on` (await their ports ≤120 s
  each) → GPU fit (need = learned `gpu_mem_gb`, free from nvidia-smi) → run `script_path`
  (cwd = script dir, stdin /dev/null, 30 s) — or, when the service has no script (e.g.
  key-manager), run the same steps itself: `new-session -d -s S -c cwd`, `send-keys` for
  `. <venv>/bin/activate` (if venv) and `<command>` — → success if session exists → await own port
  (≤120 s) → on failure return last 50 lines. Sets `should_run=1`.
- `stop(name)` — Ctrl-C, wait ≤10 s for port/pane process to go, then `kill-session`.
  Sets `should_run=0`. `restart = stop + start`.
- GPU learning: after a successful start (and on each health tick), sum nvidia-smi MB for pids
  in the pane's process tree (via `/proc` ppid table); store max as `gpu_mem_gb`.
- Health tick (existing 60 s scheduler): record up/down transitions in `service_events`.

Config changes (`service_changes` table, backups under `backups/service_changes/<id>/`):
1. `change_service_config(name, request)`: back up tracked files in the service's cwd (and the
   script) — extensions `.json .yaml .yml .env .sh .toml .cfg .ini .txt .py`, ≤1 MB, skipping
   `venv*`, `*env/`, `.git`, `node_modules`, `__pycache__`, model/data dirs (same excludes as
   the test-copy rsync) — with a manifest of (path, size, mtime, sha256).
2. Run `ask_claude_code`-equivalent (`SessionManager.run(project_for_service, prompt,
   wait=CLAUDE_WAIT)`) with prompt: "Make the smallest config change so that service <name>
   (tmux session <s>, started by <script>, command <cmd>) runs with: <request>. Do not start,
   stop or restart anything and do not call its HTTP endpoints. Say which files you changed."
   The service's cwd must belong to a registered project (auto-registered if missing).
3. Re-scan the manifest → changed/new/deleted files → unified diff (difflib) stored.
   Files outside cwd or >1 MB changed → `flagged=1`, Apply disabled.
4. Status `proposed`; chat returns a change card (diff + Apply & restart / Undo).
5. Apply (UI tap only): restart the service (and dependents) → `applied`; port check result.
   Undo: restore backups (and delete files that did not exist before) → `undone`; if it was
   applied, restart again.
- If no files changed: status `no_change` with Claude's explanation.

## 5. Pending actions (confirmation model)

Table `pending_actions(id, kind, service, payload, created_by, status, created_at, decided_at)`
with kinds `stop | restart | send | register | start_override_gpu` (config Apply is UI-only, never a pending action). Gemini tools that need
confirmation create a row and return `{"needs_confirmation": true, "action_id": …}`; the chat
UI renders Confirm / Cancel buttons; only `POST /api/actions/{id}/confirm` (authenticated
browser session) executes it. No Gemini tool can confirm. Actions expire after 15 min.
The injection guard blocks creating service actions after `web_search`/`capture_link` in the
same turn (like `ask_claude_code`).

## 6. Data model

- `services(id, name UNIQUE, script_path (nullable: services without a script are launched from
  cwd/venv/command), tmux_session UNIQUE, cwd, venv, command, port,
  depends_on TEXT (JSON list of names), gpu_mem_gb REAL, notes, managed INTEGER DEFAULT 1,
  should_run INTEGER DEFAULT 0, project_id, created_at, updated_at)`
- `service_events(id, service_id, kind, detail, created_at)` — up/down/start/stop/send/change
- `service_changes(id, service_id, request, status, claude_session, backup_dir, manifest TEXT,
  diff TEXT, flagged INTEGER, explanation, created_at, updated_at)`
- `pending_actions` (§5)

## 7. Gemini tools

`list_services()`, `service_output(name, lines=200)`, `start_service(name)`,
`stop_service(name)` → pending, `restart_service(name)` → pending,
`send_to_service(name, text)` → pending, `register_service(script_path, notes=None)` → parsed
fields for the user to confirm (pending kind `register`), `change_service_config(name,
request)`, `undo_service_change(change_id)` (undo is safe → allowed directly),
`investigate_service(name, question)` → `ask_claude_code` in the service's project with the
last 200 lines appended. Apply is UI-only.

## 8. UI — ⚙️ Services tab

- GPU bar (used/total, per-service breakdown) + service cards (●/○, port ✓/✗, GPU GB,
  session, last line, "should run but down" warning). Unmanaged sessions listed view-only.
- Detail sheet: live pane (poll `GET /api/services/{name}/output?lines=300` every 2 s, follow
  unless scrolled up), command box + Enter, Ctrl-C (confirm dialog), Start / Stop / Restart
  (confirm), Summarize (Gemini, shown inline), Investigate (opens Claude session), Change
  config (prompt → change card), change history with diffs + Undo.
- "+ Service": script path → parsed fields form → save. "Import all": scan `run_*.sh` in
  registered projects' folders → checklist → save.
- Chat: change cards and pending-action cards with buttons.

## 9. API

`GET /api/services`, `POST /api/services` (register from parsed/edited fields),
`POST /api/services/parse {script_path}`, `GET /api/services/import-candidates`,
`PATCH/DELETE /api/services/{name}`, `GET /api/services/{name}/output?lines=`,
`POST /api/services/{name}/send {text|ctrl_c}` (UI — no pending step; managed only),
`POST /api/services/{name}/start {override_gpu?}`, `…/stop`, `…/restart` (UI — direct),
`POST /api/services/{name}/summarize`, `POST /api/services/{name}/investigate {question}`,
`GET /api/services/{name}/changes`, `POST /api/changes/{id}/apply`, `POST /api/changes/{id}/undo`,
`GET /api/actions?status=pending`, `POST /api/actions/{id}/confirm`, `POST /api/actions/{id}/cancel`,
`GET /api/gpu`.

## 10. Safety

- Only registered managed sessions receive input, stops or kills; never `kill-server`.
- UI actions are direct (the user is the one tapping); Gemini-initiated risky actions go
  through pending actions (§5).
- Config backups taken before Claude runs; undo restores byte-for-byte; flagged changes
  cannot be applied.
- GPU refusals list current GPU consumers; override is a UI-only action.
- Command sends logged in `service_events` with origin (`ui` | `gemini`).

## 11. Error handling

- Start failure → last 50 lines + status `failed`; dependency failure stops the chain.
- tmux missing / no server → clear error; tab still renders registry.
- `ss` unavailable → TCP connect probe to 127.0.0.1:port.
- `nvidia-smi` unavailable → GPU "?" and fit check skipped with a warning.
- Script parse failure → Claude extraction fallback → user confirms.
- Claude change returns "still running" (CLAUDE_WAIT) → change stays `running`; the card
  updates when the Claude turn finishes (poll change status).

## 12. Testing

- Real tmux on an isolated server: fixture sets `TMUX_TMPDIR` to `mktemp -d /tmp/tx.XXXX`
  (short path) and kills that server at teardown; throwaway scripts in the user's pattern
  start `python3 -m http.server <free port> --bind 127.0.0.1`.
- Cases: parse script; register; start → alive + port; stop → gone; restart; dependency
  order (B depends on A); output capture; send text; Ctrl-C; unmanaged session refuses send;
  start failure returns tail lines; health tick records down transition.
- Pending actions: Gemini `stop_service` creates a pending row, does not stop; confirm route
  stops; expired action refused; injection guard blocks creation.
- Config change: fake Claude (session manager with fake claude script) edits `config.json`
  → backup + diff + proposed; apply restarts; undo restores byte-for-byte; edit outside cwd
  → flagged, apply refused; no-change path.
- GPU: fake `nvidia-smi` script on PATH → learned usage, refusal, override.
- Live (controller): import the user's real scripts and show status/output only; a real
  start/stop only on a throwaway demo service. The user's running services are not touched.
