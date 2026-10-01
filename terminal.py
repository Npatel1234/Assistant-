"""Web terminal: each terminal is a tmux session `webterm-N` running the login shell; a websocket attaches to it
through a pty, so closing the browser never kills what runs inside. The services tab never sees these sessions."""
import asyncio
import codecs
import contextlib
import fcntl
import os
import pty
import re
import shlex
import signal
import struct
import subprocess
import termios
from pathlib import Path

PREFIX = "webterm-"
_NAME = re.compile(r"webterm-\d{1,4}")


class TerminalError(Exception):
    pass


def valid_name(name: str) -> bool:
    return bool(_NAME.fullmatch(name or ""))


class Attach:
    """A tmux client running on a pty. close() kills only the client; the tmux session lives on."""

    def __init__(self, loop, master: int, proc):
        self.loop, self.master, self.proc = loop, master, proc
        self._q: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._closed = False
        loop.add_reader(master, self._on_readable)

    def _on_readable(self) -> None:
        try:
            data = os.read(self.master, 65536)
        except BlockingIOError:
            return
        except OSError:  # EIO: the client exited
            data = b""
        if not data:
            self.loop.remove_reader(self.master)
        self._q.put_nowait(data or None)

    async def read(self) -> str | None:
        """Next chunk of output as text; None once the client has exited."""
        while True:
            data = await self._q.get()
            if data is None:
                return self._dec.decode(b"", final=True) or None
            if text := self._dec.decode(data):
                return text

    def write(self, data: bytes) -> None:
        if self._closed:
            return
        view = memoryview(data)
        while view:
            try:
                view = view[os.write(self.master, view):]
            except BlockingIOError:  # ponytail: drop input if the pty buffer is full, no retry queue
                return
            except OSError:
                return

    def resize(self, cols: int, rows: int) -> None:
        if self._closed:
            return
        cols, rows = max(10, min(500, int(cols))), max(5, min(200, int(rows)))
        fcntl.ioctl(self.master, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))

    def close(self) -> None:
        """Synchronous on purpose: kill + reap must complete even if the caller is being cancelled."""
        if self._closed:
            return
        self._closed = True
        self.loop.remove_reader(self.master)
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(self.proc.pid, signal.SIGKILL)  # start_new_session: pgid == pid
        self.proc.wait()  # SIGKILLed, so this returns at once; no zombie
        with contextlib.suppress(OSError):
            os.close(self.master)


class TerminalManager:
    def __init__(self, tmux_env: dict | None = None, idle_s: float = 1800):
        self.env = {k: v for k, v in (os.environ if tmux_env is None else tmux_env).items() if k != "TMUX"}
        self.idle_s = idle_s
        self._lock = asyncio.Lock()

    async def _tmux(self, *args, timeout: float = 15) -> tuple[int, str]:
        try:
            proc = await asyncio.create_subprocess_exec(
                "tmux", *args, env=self.env, stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        except FileNotFoundError:
            raise TerminalError("tmux is not installed")
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout)
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()
            return 124, "tmux timed out"
        return proc.returncode, out.decode(errors="replace")

    async def list(self) -> list[dict]:
        code, out = await self._tmux("list-sessions", "-F", "#{session_name}|#{session_created}|#{session_attached}")
        if code != 0:
            if "no server running" in out or "error connecting" in out or "No such file" in out:
                return []
            raise TerminalError(f"tmux failed: {out.strip()}")
        res = []
        for line in out.splitlines():
            name, _, rest = line.partition("|")
            created, _, attached = rest.partition("|")
            if valid_name(name) and created.isdigit() and attached.isdigit():
                res.append({"name": name, "created": int(created), "attached": int(attached)})
        return sorted(res, key=lambda t: int(t["name"][len(PREFIX):]))

    async def exists(self, name: str) -> bool:
        return valid_name(name) and any(t["name"] == name for t in await self.list())

    async def create(self) -> dict:
        shell = self.env.get("SHELL") or ""
        if not (shell and os.access(shell, os.X_OK)):
            shell = "bash"
        async with self._lock:
            used = {t["name"] for t in await self.list()}
            n = next(i for i in range(1, 10000) if f"{PREFIX}{i}" not in used)
            name = f"{PREFIX}{n}"
            code, out = await self._tmux("new-session", "-d", "-s", name, "-c", str(Path.home()),
                                         "-x", "120", "-y", "32", f"{shlex.quote(shell)} -l")
            if code != 0:
                raise TerminalError(f"tmux new-session failed: {out.strip()}")
        return next(t for t in await self.list() if t["name"] == name)

    async def close(self, name: str) -> None:
        if not valid_name(name):
            raise TerminalError(f"'{name}' is not a web terminal")
        code, out = await self._tmux("kill-session", "-t", f"={name}")
        if code != 0:
            raise TerminalError(f"terminal '{name}' is not running")

    async def attach(self, name: str) -> Attach:
        if not await self.exists(name):
            raise TerminalError(f"terminal '{name}' is not running")
        master, slave = pty.openpty()
        try:
            os.set_blocking(master, False)
            fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", 32, 120, 0, 0))
            proc = subprocess.Popen(  # not asyncio's: close() must kill and reap synchronously
                ["tmux", "attach-session", "-t", f"={name}"], stdin=slave, stdout=slave, stderr=slave,
                start_new_session=True, env={**self.env, "TERM": "xterm-256color"})
        except BaseException:
            os.close(master)
            raise
        finally:
            os.close(slave)
        return Attach(asyncio.get_running_loop(), master, proc)
