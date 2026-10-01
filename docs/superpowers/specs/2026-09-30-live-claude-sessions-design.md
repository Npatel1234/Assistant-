# Live Claude Sessions + Resume After Usage Reset — Design Spec

Date: 2026-09-30
Status: Draft for review
Builds on: `2026-09-29-gemini-assistant-design.md` (the assistant as merged on master)
Part 1 of 3 (next: #2 service manager / tmux, #3 GitHub workspace coder + tester)

## 1. Goal

Every Claude Code run the assistant starts becomes a **live session** the user can watch in the UI
as it happens, type into (follow-up turns in the same Claude conversation), and stop. When Claude
hits its usage limit, the session pauses with the exact reset time (shown in IST) and the user can
have it **resume automatically when usage resets**.

### Success criteria
- Asking Gemini a code question shows a "watch live" link; the transcript streams Claude's text and
  tool calls within ~1 s of Claude emitting them; Gemini still answers from Claude's result.
- The user can send a follow-up message into a running or idle session from the UI and see the reply.
- Stop kills the session's whole process group.
- Up to 3 sessions run in parallel, never two on the same project; extra requests queue visibly.
- On a usage-limit rejection the session shows "⏸ paused · resets HH:MM IST"; "resume at reset"
  (button or Gemini tool) re-sends the pending message after the reset, surviving an app restart.
- UI shows Claude usage (5-hour and 7-day utilization + reset times in IST).

## 2. Verified CLI behaviour (Claude Code 2.1.284, 2026-09-30)

- `claude -p --input-format stream-json --output-format stream-json --verbose
  --permission-mode bypassPermissions [--resume <id>]` keeps one process alive across turns.
  Each user turn is one stdin line:
  `{"type":"user","message":{"role":"user","content":"<text>"}}`.
- Stdout is one JSON event per line. Observed types: `system` (subtypes `hook_started`,
  `hook_response`, `init`, `commands_changed`), `assistant` (`message.content[]` of `text` /
  `tool_use` / `thinking`), `user` (tool results), `rate_limit_event`, `result`.
- A turn ends with `type:"result"` carrying `is_error`, `result` (final text), `session_id`,
  `api_error_status`, `num_turns`, `duration_ms`. Context carries across turns in one process.
- `rate_limit_event.rate_limit_info` = `{status: "allowed"|…, resetsAt: <epoch s>,
  rateLimitType: "five_hour"|…, unifiedWindows: {five_hour: {utilization, resetsAt},
  seven_day: {utilization, resetsAt}}}`.
- Closing stdin ends the process cleanly (exit 0).

## 3. Scope

In: session manager, streaming transcript UI, send/stop, parallel limit + queue, idle close +
resume, usage-limit pause + scheduled resume, usage meter, clock sanity check.
Out (later parts): tmux/service management (#2), git workspace/tester (#3), real terminal view,
interactive permission prompts (sessions keep full access per original spec).

## 4. Architecture

```
Gemini chat ──ask_claude_code──┐            UI: Claude tab + chat tool block
                               ▼               ▲ SSE /api/claude/sessions/{id}/stream
                    SessionManager (in-process)│ POST .../send  .../stop  .../resume-at-reset
                    ├─ max 3 running, 1 per project, FIFO queue beyond
                    ├─ Session: subprocess (own process group) + stdin writer + stdout reader task
                    ├─ events → SQLite + fan-out to live watchers (asyncio queues)
                    ├─ idle 30 min after last result → close stdin (status "idle-closed")
                    └─ scheduler tick: due scheduled resumes → reopen with --resume
```

Replaces `claude_runner.ClaudeRunner` usage in `tools.ask_claude_code`. The old one-at-a-time
runner and `claude_runs` table stay for history; new runs also write a `claude_runs` row per
completed turn (so the project page and Gemini context keep working).

### Files
- `claude_sessions.py` (new): `SessionManager`, `Session`, event parsing, limit detection.
- `tools.py`: `ask_claude_code` uses the manager; new tools `list_claude_sessions`,
  `schedule_claude_resume`.
- `jobs.py`: scheduler tick fires due resumes; clock check.
- `main.py`: session API routes + SSE.
- `static/index.html`: Claude tab, transcript view, usage meter, chat "watch live" link.
- `db.py`: new tables.

## 5. Data model

- `claude_sessions(id, project_id, claude_session_id, status, pending_message, resume_at,
  resume_estimated INTEGER, resume_scheduled INTEGER DEFAULT 0, last_error, created_at, updated_at, last_activity_at)`
  - `status`: `queued | starting | running | idle | closed | paused_limit | failed | stopped`
- `claude_events(id, session_id, seq, kind, payload TEXT, created_at)` — every stdout event plus
  user-sent turns (`kind="user_turn"`). Payload capped at 64 KB per event (truncated marker).
- `kv['claude_usage']` = latest `rate_limit_info` JSON; `kv['clock_offset_s']`.

## 6. Session lifecycle

1. **Open** (`ask_claude_code`, UI send to a closed session, or scheduled resume): if the
   project already has a live session (starting/running/idle) and the request is a follow-up,
   reuse it. A non-follow-up request while that session is `idle` closes it first and opens a
   fresh conversation (no `--resume`); while it is `running`, the request queues until it goes
   idle. Otherwise create a row `queued`. Start when running count < 3 and no live session on that project.
2. **Start**: spawn process (cwd = project path, `start_new_session=True`, `CLAUDECODE` removed,
   `--resume <claude_session_id>` when the row has one). Status `running` while a turn is in
   flight, `idle` after its `result`.
3. **Turn**: write the user line to stdin; record `user_turn` event. The reader task stores and
   fans out each stdout event; on `result` it completes that turn's future (Gemini awaits it for
   `ask_claude_code`, with the existing `CLAUDE_TIMEOUT`), writes a `claude_runs` row
   (`files_changed` via git status before/after the turn as today), stores `claude_session_id`.
   Turns sent while a turn is running are queued in-order by the session (one in flight).
4. **Idle close**: 30 min (`CLAUDE_IDLE_MIN`) after the last `result` with no new turn → close
   stdin, wait, status `closed`. A later turn reopens with `--resume`.
5. **Stop** (UI or app shutdown): kill process group (SIGKILL), status `stopped`; any awaiting
   Gemini call gets `{"error": "stopped by user"}`.
6. **Crash / exit without result**: status `failed`, `last_error` = last 2 KB stderr; resumable.
7. **Timeout** of a turn: kill group, status `failed` with "timed out"; resumable.

## 7. Usage limits and resume at reset

- Every `rate_limit_event` updates `kv['claude_usage']`.
- A turn is a **limit rejection** when either: the latest `rate_limit_event` in that turn has
  `status != "allowed"`, or the `result` has `is_error` and its text matches
  `/usage limit|limit reached|rate limit/i`.
- Reset time: `rate_limit_info.resetsAt` (epoch) → else text `|<epoch>` → else text
  `resets <h>(:mm)?(am|pm)( \((TZ)\))?` interpreted in the stated TZ (default UTC, the machine TZ)
  as the next such time → else now + 5 h with `resume_estimated=1`.
- Session → `paused_limit`, `pending_message` = the rejected turn's text, `resume_at` stored
  (UTC). Process closed. Gemini's awaiting call gets `{"error": "Claude usage limit reached;
  resets <HH:MM IST>. Ask the user whether to resume then."}`; greeting mentions paused sessions.
- **Schedule**: UI button or `schedule_claude_resume(project, message?, at?)` sets
  `resume_scheduled=1` (column) with optional replacement message and optional time (default
  `resume_at + 2 min`). Scheduler tick (existing 60 s loop) fires due ones: reopen with
  `--resume`, send the message. If rejected again → pause again with the new reset, no loop.
- All times stored UTC; displayed via `config.TZ`.

## 8. Clock sanity

Machine clock verified accurate (UTC timezone). Safety net: on each Gemini HTTP response, read
the `Date` header when available; store `clock_offset_s = server − local`. If |offset| > 120 s,
log a warning, show it in `/api/status`, and add the offset when computing "now" for resume
scheduling. (If the header isn't reachable through the SDK, compute it from a lightweight HEAD
request to `https://generativelanguage.googleapis.com` once per hour.)

## 9. API

- `GET /api/claude/sessions` → list (id, project, status, resume_at, pending, updated).
- `GET /api/claude/sessions/{id}/events?after=<seq>` → stored events.
- `GET /api/claude/sessions/{id}/stream` → SSE: replay after `seq`, then live.
- `POST /api/claude/sessions/{id}/send {text}` → queue a turn (reopens if closed/failed).
- `POST /api/claude/sessions/{id}/stop`.
- `POST /api/claude/sessions/{id}/resume-at-reset {message?}`; `DELETE` same path cancels.
- `GET /api/claude/usage` → latest usage windows with reset times.

## 10. UI

- New **Claude** tab: session cards (project, status chip, last activity; paused cards show
  "resets 19:00 IST" + **Resume at reset** / cancel). Usage meter at top (5-hour and 7-day bars,
  reset times IST).
- Transcript view (full-screen sheet): 🤖 text blocks (markdown), 🔧 one-line tool summaries
  (`Read path`, `Edit path`, `Bash cmd` → tap to expand input/output), result footer (duration,
  files changed). Composer + Send, Stop button, status chip. Auto-scroll unless the user scrolled up.
- Chat tool block for `ask_claude_code` gets a **watch live** link to that session.

## 11. Gemini tools

- `ask_claude_code(project, prompt, follow_up=False)` — unchanged signature; `follow_up=True`
  means continue the project's session (live or via resume), else a fresh Claude conversation.
  Result adds `session_id` (our id) for the UI link.
- `list_claude_sessions()` — statuses, paused reset times (IST).
- `schedule_claude_resume(project, message=None, at=None)` — `at` ISO local, default reset + 2 min.
- The injection guard (no `ask_claude_code` after web content in the same turn) still applies;
  it also applies to `schedule_claude_resume`.

## 12. Error handling & limits

- Queue full → `ask_claude_code` waits (Gemini told "queued, position N" via a tool event).
- Event storage cap: 5,000 events per session; older `assistant`/`user` payloads beyond that are
  truncated to 1 KB (UI shows "truncated").
- Unknown event types stored and shown as a muted line (forward compatible).
- Sessions are in-process: app restart stops running ones (status `stopped`), keeps
  `paused_limit` + scheduled resumes.

## 13. Testing

Fake `claude` executable (Python script) that reads stream-json turns from stdin and emits
scripted events per prompt keyword, covering:
- normal single turn; two turns in one process (context via counter); `--resume` arg recorded
- follow-up send while running is queued in order
- stop kills group (grandchild dies)
- idle close after a short test idle timeout; later turn reopens with `--resume`
- parallel cap: 4th session queued, starts when one finishes; same-project reuse
- limit rejection via `rate_limit_event status=rejected` → paused with exact `resetsAt`
- limit via text `…|<epoch>` and via "resets 7pm (UTC)"; unparseable → +5 h estimated
- scheduled resume fires at due time (injected clock), re-pauses on second rejection
- crash without result → failed; turn timeout → failed
- SSE endpoint replays stored events then streams new ones; auth required
Plus one real `claude` smoke run and a manual UI check.
