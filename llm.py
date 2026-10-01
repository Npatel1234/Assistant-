"""Gemini wrapper: ordered model fallback, local RPM throttle, quota bookkeeping, embeddings."""
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
from google.genai import errors, types

import db

PACIFIC = ZoneInfo("America/Los_Angeles")  # Gemini daily quotas reset at midnight Pacific


class AllModelsExhausted(Exception):
    def __init__(self, until: float):
        super().__init__(f"all models out of quota until {until}")
        self.until = until


def next_pacific_midnight(ts: float) -> float:
    d = datetime.fromtimestamp(ts, PACIFIC)
    return (datetime(d.year, d.month, d.day, tzinfo=PACIFIC) + timedelta(days=1)).timestamp()


def classify(err: errors.APIError) -> str:
    if err.code == 429:
        return "day" if "PerDay" in str(err) else "minute"
    if err.code == 404:
        return "dead"
    if err.code >= 500:
        return "minute"  # 503 overloaded etc: skip this model for a minute
    return "fatal"


def pack(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def unpack(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def top_k(query: np.ndarray, items: list[tuple], k: int) -> list[tuple]:
    """items: [(key, packed_vector)] -> [(key, cosine)] best first."""
    if not items:
        return []
    mat = np.stack([unpack(b) for _, b in items])
    sims = mat @ query / (np.linalg.norm(mat, axis=1) * np.linalg.norm(query) + 1e-9)
    return [(items[i][0], float(sims[i])) for i in np.argsort(-sims)[:k]]


class LLM:
    def __init__(self, client, con, models, embed_model, search_models, rpm_limit, clock=time.time):
        self.client, self.con = client, con
        self.models, self.embed_model, self.search_models = models, embed_model, search_models
        self.rpm_limit, self.clock = rpm_limit, clock
        self.calls: dict[str, deque] = defaultdict(deque)
        self.dead: set[str] = set()
        self.last_model: str | None = None

    def exhausted_until(self, model: str) -> float:
        r = db.one(self.con, "SELECT exhausted_until FROM model_state WHERE model=?", model)
        return r["exhausted_until"] if r else 0.0

    def _mark(self, model: str, until: float) -> None:
        self.con.execute("INSERT INTO model_state(model, exhausted_until) VALUES(?, ?) "
                         "ON CONFLICT(model) DO UPDATE SET exhausted_until=excluded.exhausted_until",
                         (model, until))

    def _usable(self, model: str) -> bool:
        now = self.clock()
        if model in self.dead or self.exhausted_until(model) > now:
            return False
        q = self.calls[model]
        while q and q[0] <= now - 60:
            q.popleft()
        # ponytail: at the local RPM cap we move to the next model instead of waiting
        return len(q) < self.rpm_limit

    def _fail(self, model: str, err: errors.APIError) -> None:
        kind = classify(err)
        if kind == "fatal":
            raise err
        if kind == "dead":
            self.dead.add(model)
        elif kind == "day":
            self._mark(model, next_pacific_midnight(self.clock()))
        else:
            self._mark(model, self.clock() + 60)

    async def _call(self, chain: list[str], fn):
        for model in chain:
            if not self._usable(model):
                continue
            self.calls[model].append(self.clock())
            try:
                return await fn(model), model
            except errors.APIError as e:
                self._fail(model, e)
        now = self.clock()
        until = min((self.exhausted_until(m) for m in chain if m not in self.dead), default=0.0)
        raise AllModelsExhausted(until if until > now else now + 60)

    async def generate(self, contents, config=None, background=False):
        chain = self.models[:1] if background else self.models  # jobs never eat the chat quota
        resp, model = await self._call(chain, lambda m: self.client.aio.models.generate_content(
            model=m, contents=contents, config=config))
        if not background:
            self.last_model = model
        return resp, model

    async def search(self, query: str):
        cfg = types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())])
        return await self._call(self.search_models, lambda m: self.client.aio.models.generate_content(
            model=m, contents=query, config=cfg))

    async def embed(self, texts: list[str]) -> list[np.ndarray] | None:
        """None when the embedding model is unavailable; callers fall back to keywords."""
        try:
            resp, _ = await self._call([self.embed_model], lambda m: self.client.aio.models.embed_content(
                model=m, contents=texts))
        except AllModelsExhausted:
            return None
        return [np.asarray(e.values, dtype=np.float32) for e in resp.embeddings]
