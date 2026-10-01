# Security

This is a single-user tool that deliberately has a lot of power on the machine it runs on. Read this before running it anywhere other than a trusted home or lab network.

## Threat model

- **One user, trusted LAN.** The app has one password and no user accounts. It is designed for you, on your own network.
- **Plain HTTP.** The server speaks HTTP only and listens on all interfaces (`0.0.0.0`). Anyone who can sniff the network can read traffic and steal the session cookie (the cookie is marked `Secure` only when the request itself is HTTPS).
- **Do not expose it to the internet.** Tunnels such as ngrok were out of scope in the design. If you tunnel it anyway, you are putting a password-protected remote shell and full-access coding agent on the public internet; use HTTPS, a very long password, and preferably an additional access layer (VPN, SSO proxy, IP allow list).
- **Out of scope:** multiple users, multi-tenant use, defending against someone who already has a shell as the same OS user.

## Authentication

- Login with `APP_PASSWORD`; success sets a signed cookie (`itsdangerous`, keyed by `SESSION_SECRET`), `HttpOnly`, `SameSite=Lax`, valid 30 days. All routes except `/login` require it; APIs answer 401, pages redirect to `/login`.
- Password comparison is constant-time. Failed attempts are limited per client IP: 5 failures in 15 minutes returns 429 until the window passes. The same limiter covers the terminal unlock.
- Changing `SESSION_SECRET` invalidates every existing login. The app refuses to start if the password, secret or Gemini key is empty.
- The limiter is in memory and keyed by IP; behind a reverse proxy all clients may appear as one IP.

## What the app can do

Treat a logged-in session as shell access to the host account running the app.

- **Claude Code runs with full access** (`--permission-mode bypassPermissions`) inside registered project folders, with no per-action prompts. It can read, write and execute anything that user can.
- **Web terminal:** a real login shell as the same user (needs the extra unlock below).
- **Service manager:** types commands into tmux sessions, starts and stops processes, edits service files through Claude (with backups).
- **Repo workspace:** clones repos, runs their test code, can `git push` with your git credentials (never force) and, after confirmation, `pip install` into a venv you designated.
- **Gemini** sees whatever the tools return: project notes, memories, tool output, web pages you save, service logs. Data is sent to Google's API.

## Prompt-injection guard

The assistant reads untrusted text (web pages from `capture_link`, `web_search` results, service/terminal output, repo contents). A malicious page could try to instruct the model to run tools. Mitigations:

- The system prompt tells the model that tool results, memories and synced notes are data, never instructions, and that Claude Code may only be called for what the user asked in their own message.
- **Taint tracking (enforced in code, not by the model):** once a turn has used a tool that returns untrusted content (repo tools, `capture_link`, `web_search`, `service_output`, `investigate_service`, `change_service_config`, `call_service`), the guarded tools (`ask_claude_code`, `schedule_claude_resume`, `change_service_config`, `investigate_service`, `add_repo`, `build_test_suite`, `start_repo_task`, `fix_task_regressions`, `pull_repo`, `discard_task`) are refused for the rest of that turn, and `start_service` becomes a pending Confirm action. The UI shows an **Allow** button; tapping it is you authorizing that call in a new message.
- This reduces, but does not eliminate, risk: the guard is per turn, and memories or notes written earlier from untrusted data are not re-tainted. Do not save links or content you do not trust.

## Pending actions: the Confirm model

Stop, restart, send input, register service, start with GPU override, non-GET HTTP calls to services, repo merge, undo merge, push and package install are never executed by the model. The tool only creates a `pending_actions` row. It runs only when you tap **Confirm** (an authenticated `POST /api/actions/{id}/confirm`). No tool can confirm. Actions expire after 15 minutes and cannot be confirmed twice. Typing "confirm" in chat does nothing.

## Web terminal

- Requires the login cookie **and** a separate unlock with the app password (rate limited as above). The unlock lives in server memory, expires after `TERMINAL_IDLE_MIN` minutes without typing, and is cleared by Lock, logout and restart.
- The WebSocket route repeats all checks before accepting: valid login cookie, current unlock, a valid `webterm-N` name that exists, and an `Origin` header whose host equals the request `Host` (blocks cross-site WebSocket hijacking). Missing Origin is refused. An open socket is closed within about 2 seconds of locking.
- Gemini has no tool to read or type into terminals; `webterm-*` sessions are hidden from the service tools.

## Secrets

- `.env` holds `GEMINI_API_KEY`, `APP_PASSWORD` and `SESSION_SECRET`. It is listed in `.gitignore` and must never be committed. Only `.env.example` (empty secrets) belongs in git. `.env.example` is not read by the app.
- Generate the secret with `python -c "import secrets; print(secrets.token_urlsafe(32))"`. Use a long unique password.
- `assistant.db` (conversations, memories, Claude transcripts, repo and service data), `assistant.log` and `backups/` are git-ignored. They contain personal data and, for service-change backups, possibly copies of `.env` files from your services. Protect them like secrets and do not publish them.
- Do not paste secrets into chat: messages are stored and sent to Gemini.
- If a key or password leaks, rotate it: revoke the Gemini key, change `APP_PASSWORD`, change `SESSION_SECRET`.

## Recommendations

- Run on a private network behind a firewall; allow the port only for your devices.
- Use a dedicated low-privilege OS user if you can, and keep valuable credentials (SSH keys, cloud tokens) out of its reach. Remember that it can run Claude Code and a shell.
- Prefer HTTPS via a reverse proxy or VPN for anything beyond a home LAN.
- Review the diff before **Apply & restart**, and review a repo test suite before **Accept suite**.
- Keep the Claude Code CLI and dependencies up to date (`uv sync`).

## Reporting a vulnerability

Please report security issues privately rather than in a public issue: use GitHub's private vulnerability reporting on this repository (Security tab, "Report a vulnerability"), or contact the maintainer, Npatel1234, through their GitHub profile. Include the affected version or commit, steps to reproduce and the impact. This is a personal project maintained on a best-effort basis; there is no formal response time.
