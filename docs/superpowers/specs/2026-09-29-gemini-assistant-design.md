# Gemini Personal Assistant — Design Spec

Date: 2026-09-29
Status: Draft for review

## 1. Goal

A personal assistant web app, powered by Gemini (free tier), that tracks the user's
projects and ideas with short- and long-term memory, feels proactive/agentic rather
than a plain chatbot, and can delegate code questions/tasks to Claude Code (Pro plan)
running in the user's project folders.

Single user. Runs on this Linux machine, reachable from laptop and phone on the home
network (LAN). Remote access (ngrok) is out of scope for v1.

### Success criteria
- Open app on phone or laptop browser on LAN, log in, chat with assistant.
- Assistant remembers facts/ideas/projects across sessions and uses them in answers.
- Projects are seeded and kept current from Claude Code history automatically.
- Asking "in ai call bot, which providers do I have?" launches Claude Code in that
  project folder and returns the answer in chat.
- Hitting a Gemini quota limit switches models automatically; chat keeps working
  until all models are exhausted.

## 2. Scope

### In v1
- Chat with streaming replies, tool use, proactive greeting
- Memory: short-term (conversation), long-term (memories + structured records), profile
- Projects, ideas inbox, idea → project promote
- Claude Code history auto-sync (every 3h)
- Claude Code delegation (`ask_claude_code`) in registered project folders, full access
- Reminders (dashboard + greeting), daily/weekly recap
- Decision log per project
- Web research (Gemini Google Search grounding)
- Link capture (fetch URL → summarize → file under project/idea)
- Resume-context button (paste-ready Claude Code prompt per project)
- "What should I work on now?"
- Model fallback chain with RPM throttle + RPM/RPD 429 handling
- Password login
- Mobile-first single-page UI

### Out of v1 (later)
ngrok/internet access, idea critic, PWA install, Telegram/push notifications,
voice input, multi-user, file uploads.

## 3. Architecture

```
Phone/Laptop browser ──(LAN)──► FastAPI (single process, uvicorn, 0.0.0.0:PORT)
                                 ├─ /login               password → signed session cookie
                                 ├─ /api/chat            Gemini + tools, SSE streaming
                                 ├─ /api/*               projects, ideas, memories, reminders,
                                 │                       decisions, recaps, claude runs
                                 ├─ static/index.html    mobile-first SPA (vanilla JS, no build)
                                 └─ background loop (asyncio tasks)
                                     ├─ every 1 min : due reminders
                                     ├─ every 3 h   : Claude history sync
                                     └─ nightly     : reflection + daily recap (+ weekly on Sunday)
                                              + DB backup
                       SQLite assistant.db (WAL) ◄──┘
                       Gemini API via google-genai SDK (fallback chain)
                       Claude Code CLI subprocess (headless)
```

Stack: Python 3, FastAPI, uvicorn, `google-genai`, `httpx`, `bcrypt`, `itsdangerous`,
stdlib `sqlite3`. Run with `uv run main.py`. Config in `.env`.

File layout (keep few files):
- `main.py` — FastAPI app, routes, auth, background loop startup
- `db.py` — schema + queries
- `llm.py` — Gemini client, fallback chain, throttle, embeddings
- `agent.py` — prompt building, tool definitions + dispatch, memory extraction
- `jobs.py` — Claude history sync, reflection, recap, reminders, backup
- `claude_runner.py` — Claude Code subprocess delegation
- `static/index.html` — UI (inline CSS/JS)
- `test_core.py` — core checks

## 4. Data model (SQLite)

- `messages(id, conversation_id, role, content, created_at)`
- `conversations(id, title, summary, created_at, updated_at)` — `summary` = rolling
  summary of older turns
- `memories(id, text, kind[fact|preference|decision|note], source[chat|claude-sync|manual|link],
  project_id NULL, embedding BLOB NULL, created_at, updated_at)`
- `projects(id, name UNIQUE, path NULL, main_files NULL, goal, status[active|stalled|done|paused],
  next_step, notes, last_activity_at, claude_session_id NULL, created_at)`
- `ideas(id, text, tags, status[inbox|promoted|dropped], project_id NULL, embedding BLOB NULL, created_at)`
- `reminders(id, text, due_at, done, project_id NULL, created_at)`
- `decisions(id, project_id, text, why, source, decided_at)`
- `links(id, url, title, summary, project_id NULL, idea_id NULL, created_at)`
- `recaps(id, period[daily|weekly], period_start, content, created_at)`
- `profile(id=1, content, updated_at)`
- `claude_runs(id, project_id, prompt, output, session_id, files_changed, exit_code,
  duration_s, created_at)`
- `sync_state(file_path PRIMARY KEY, sha256, synced_at)`
- `model_state(model PRIMARY KEY, exhausted_until)`

Embeddings stored as float32 blobs; similarity = brute-force cosine over all rows
(fine up to tens of thousands of rows).

## 5. Memory

| Layer | Content | Lifetime |
|---|---|---|
| Short-term | last ~20 messages + rolling summary of older turns | per conversation |
| Long-term memories | facts/preferences/decisions with embeddings | forever, editable |
| Structured | projects, ideas, reminders, decisions, links | forever |
| Profile | short summary of the user, rewritten nightly | forever, editable |

Per chat turn:
1. Embed user message.
2. Retrieve top 8 memories by cosine similarity (keyword LIKE fallback if embeddings
   unavailable), active projects (name/status/next_step), due reminders, profile.
3. Build system prompt + short-term history → Gemini with tools.
4. Stream answer; execute tool calls as they arrive (max 6 tool rounds per turn).
5. After reply: one cheap extraction call returns new memories as JSON; each is
   embedded; if cosine ≥ 0.9 with an existing memory → update that one, else insert.

UI shows a "🧠 N memories" chip on each reply, tappable to list the memories used.

## 6. Agent tools

- `save_memory(text, kind, project?)`, `search_memory(query)`
- `create_project(name, goal?, path?, main_files?)`, `update_project(name, fields…)`,
  `list_projects(status?)`
- `save_idea(text, tags?)`, `promote_idea(id)` — creates project, drafts goal + 3–5 first
  steps (stored as next_step + notes)
- `set_reminder(text, when)` — Gemini converts natural language to ISO datetime using
  the current local time given in the system prompt; server validates it parses and
  is in the future. `list_reminders()`, `complete_reminder(id)`
- `log_decision(project, text, why)`, `list_decisions(project)`
- `capture_link(url, project?)` — httpx fetch (10s timeout, 2 MB cap), strip HTML to
  text, Gemini summary, store
- `get_recap(period)`
- `what_to_work_on()` — ranks projects by due reminders, days stalled, recent momentum;
  Gemini explains top 3
- `resume_context(project)` — builds paste-ready Claude Code prompt from project goal,
  status, next step, recent decisions, relevant memories
- `ask_claude_code(project, prompt)` — see §8
- Web research: Gemini's built-in Google Search grounding enabled on chat requests
  (if the SDK/model disallows combining search grounding with function tools, expose
  a `web_search(query)` tool that makes a separate grounded call instead).

Invalid tool args → error string returned to Gemini so it can correct itself.

## 7. Proactive behavior

- **Greeting**: opening a new conversation triggers an assistant-first message built from
  due reminders, stalled projects (no activity > 7 days), latest recap, and new Claude
  sync updates since last visit.
- **Related items**: when an idea or memory is saved, top 3 similar ideas/projects
  (cosine ≥ 0.75) are shown and mentioned.
- **Nightly reflection** (02:00 local): review the day's messages → merge near-duplicate
  memories, update project statuses (active → stalled after 7 days idle), rewrite
  profile, write daily recap; on Sunday also weekly recap.

## 8. Claude Code integration

### 8.1 History sync (every 3 h + on startup)
- Scan `~/.claude/projects/*/*.jsonl`.
- For each file whose sha256 differs from `sync_state`: extract user prompts and the
  assistant's final text messages of each turn (skip tool calls/results), truncate to
  ~30k chars keeping the most recent part.
- Map folder name (e.g. `-mnt-newdisk-neel-key-manager`) to project; create project if
  new with `path` decoded from the folder name when that directory exists.
- Gemini (cheapest model) returns JSON: goal, status, next_step, notes, decisions[],
  memories[]; merge into DB. Update `last_activity_at` from latest transcript timestamp.
- Bad/unparseable file → log, skip, continue. Per-run cap on Gemini calls
  (`SYNC_MAX_CALLS`, default 30) so sync can't drain the chat quota; remaining files
  wait for the next run.

### 8.2 Delegation (`ask_claude_code`)
- Project must be registered with an existing `path`; otherwise tool returns an error
  asking the user for the path.
- Command (flags verified against installed `claude --help`, v2.1.284):
  `claude -p "<prompt>" --output-format json --permission-mode bypassPermissions`
  plus `--resume <claude_session_id>` when the project has a stored session and the
  request is a follow-up. `cwd` = project path.
- Full access per user decision. Safety rails:
  - runs only in registered project folders
  - login required for all routes
  - one Claude run at a time (asyncio lock); others wait; UI shows queued/working
  - timeout `CLAUDE_TIMEOUT` (default 600 s) → process killed, partial result reported
  - if path is a git repo: `git status --porcelain` before/after → `files_changed`
  - every run stored in `claude_runs`, viewable in the project detail page
- Parse JSON result → `result` text + `session_id` (saved on project). Result returned to
  Gemini, which answers the user. UI shows a "🔧 Claude Code in <project>…" status
  block with expandable raw output.

## 9. Gemini quota handling (`llm.py`)

- `MODELS` in `.env`: ordered list. Default (verified 2026-09-29 against this API key):
  `gemini-3.5-flash-lite, gemini-3.1-flash-lite, gemini-3.5-flash, gemini-2.5-flash`.
  (`gemini-2.5-flash-lite` returns 404 for new users — excluded.)
- Per-model client-side throttle: sliding 60 s window, limit `RPM_LIMIT` per model
  (configurable per model); waits if under a few seconds, else moves to next model.
- On 429 / `RESOURCE_EXHAUSTED`:
  - error details indicate per-day quota → `exhausted_until` = next midnight
    US/Pacific
  - otherwise (per-minute) → `exhausted_until` = now + 60 s
  - retry immediately on next non-exhausted model
- On 503 (model overloaded) → `exhausted_until` = now + 60 s, retry on next model.
- 404 (model retired) → skip for the rest of the process lifetime, log a warning.
- All exhausted → chat shows "All models out of quota until HH:MM"; background jobs
  skip and retry next cycle.
- Background jobs use the cheapest model only and have per-run call caps.
- Embeddings: `EMBED_MODEL` (default `gemini-embedding-2`, 3072-dim; fallback
  `gemini-embedding-001`) with same exhaustion handling; unavailable → keyword search
  fallback, embedding backfilled later.
- Other Gemini errors: retry once, then show error in chat; user message is kept.
- Header shows the model that answered the last message.

## 10. Security

- `APP_PASSWORD` in `.env` → bcrypt hash at startup; login rate limit 5 failures / IP /
  15 min.
- Session: `itsdangerous`-signed HttpOnly SameSite=Lax cookie, 30 days, `SESSION_SECRET`
  in `.env`.
- All routes except `/login` and its static assets require a session.
- Gemini API key stays server-side.
- Binds `0.0.0.0` for LAN access.

## 11. UI

Single `static/index.html`, vanilla JS, mobile-first; bottom tab bar on phone, sidebar
on desktop.

1. **Chat** — streaming, proactive greeting, memory chip, Claude run blocks, model
   indicator, conversation list.
2. **Projects** — cards (status, next step, days idle); detail view: edit fields, path,
   decisions timeline, links, Claude runs, **Resume context** button (copy to clipboard).
3. **Ideas** — inbox, quick add, related items, Promote button.
4. **Memory** — search/edit/delete memories; edit profile.
5. **Today** — due reminders, "What to work on now", latest daily/weekly recap.

## 12. Time handling

Machine timezone is UTC and must not be changed. All times stored in UTC; code uses
`zoneinfo.ZoneInfo(TIMEZONE)` (default `Asia/Kolkata`, set in code, overridable via
`.env`) for: current time in the system prompt, parsing reminder `when`, nightly job
schedule (02:00 IST), recap day boundaries, and all UI display. Never rely on
the OS local time.

## 13. Error handling & ops

- SQLite WAL mode; nightly backup to `backups/assistant-YYYY-MM-DD.db`, keep last 7.
- Background job exceptions logged, never crash the server.
- Logs to stdout + `assistant.log`.
- Optional systemd user unit for start on boot (documented in README).

## 14. Testing

`test_core.py`, fake Gemini client, runnable with `uv run pytest` (or plain asserts):
- fallback: per-minute 429 → next model, 60 s exhaustion; per-day 429 → until reset;
  503 → next model
- memory merge: near-duplicate updates instead of inserting
- sync: unchanged hash skipped; changed file processed; bad file skipped
- reminder: `when` validation rejects past/unparseable times; IST↔UTC conversion correct
- login rate limit after 5 failures
- claude_runner: builds correct command/cwd, rejects unregistered project, timeout kill
  (with a fake `claude` script)

UI checked manually at phone width (~390 px) and desktop.

## 15. Config (`.env`)

```
gemini_api_key=          # read case-insensitively
MODELS=gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-3.5-flash,gemini-2.5-flash
EMBED_MODEL=gemini-embedding-2
RPM_LIMIT=15
APP_PASSWORD=
SESSION_SECRET=
PORT=8765
TIMEZONE=Asia/Kolkata
CLAUDE_TIMEOUT=600
SYNC_MAX_CALLS=30
```

## 16. Revisions from build-time verification (2026-09-29)

Verified live against this machine's API key and Claude CLI; these override earlier sections.

- **Web search**: Google Search grounding returns 429 on free tier for `gemini-3.5-flash-lite`
  and `gemini-3.1-flash-lite`; works on `gemini-2.5-flash`. So web search is a separate
  `web_search(query)` tool using its own chain `SEARCH_MODELS=gemini-2.5-flash`.
- **Embeddings**: single `EMBED_MODEL=gemini-embedding-2`. No fallback to
  `gemini-embedding-001` — different models' vectors are not comparable. Unavailable →
  keyword search; NULL embeddings backfilled nightly.
- **Streaming**: SSE streams events (tool calls, tool results, final answer), not
  tokens. Keeps Gemini 3 tool-call thought signatures intact by appending whole
  `candidates[0].content`. Chat turn runs as a detached task so a phone locking mid-turn
  does not cancel a running Claude Code job; answer is saved and visible on reload.
- **Password**: compared with `hmac.compare_digest` (plaintext lives in `.env` anyway,
  bcrypt adds nothing). No bcrypt dependency.
- **SDK retries** disabled (`HttpRetryOptions(attempts=1)`); our fallback chain handles
  retries.
- **Local RPM throttle**: at the per-model cap the call moves to the next model instead
  of waiting.
- **Reminders**: no 1-minute job; dashboard/greeting query due reminders directly. Browser
  notifications dropped (require HTTPS; LAN is plain HTTP). Today-tab badge polls every
  60 s.
- **Profile**: stored in `kv` table (key `profile`) instead of its own table.
- **Claude Code**: prompt passed on stdin (safe for prompts starting with `-`);
  `--output-format json` returns `result` + `session_id`. Empty stdout with exit 0 is
  reported as an error (seen when network is blocked). `CLAUDECODE` env var stripped so
  nesting inside a Claude session works.
- **History sync**: project path taken from each transcript line's `cwd` field (folder
  names are lossy: `-mnt-newdisk-neel-key-manager` ↔ `/mnt/newdisk/neel/key_manager`).
  If a run is cut short by quota/cap, next sync runs in 10 min instead of 3 h, so the
  initial import of 124 sessions finishes on its own.
- **Files**: `config.py`, `memory.py`, `tools.py` added; `agent.py` keeps prompt/loop/
  greeting/extraction only.
