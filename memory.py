"""Long-term memory: dedup-merging writes, vector recall with keyword fallback, related items."""
import re

import numpy as np

import db
from llm import pack, top_k

DUP = 0.9          # cosine at/above which two memories are "the same" -> merge
RELATED = 0.75     # threshold for "this relates to that" suggestions
RECALL_MIN = 0.55  # ponytail: tuning knob; raise if recall pulls in noise, lower if it misses things


def _vectors(con, table: str) -> list[tuple]:
    return [(r["id"], r["embedding"]) for r in
            db.rows(con, f"SELECT id, embedding FROM {table} WHERE embedding IS NOT NULL")]


async def remember(con, llm, text, kind="note", source="chat", project_id=None) -> dict:
    vecs = await llm.embed([text])
    vec = vecs[0] if vecs else None
    if vec is not None:
        hit = top_k(vec, _vectors(con, "memories"), 1)
        if hit and hit[0][1] >= DUP:
            mid = hit[0][0]
            db.update(con, "memories", mid, text=text, kind=kind, embedding=pack(vec), updated_at=db.now())
            return {"id": mid, "merged": True}
    mid = db.insert(con, "memories", text=text, kind=kind, source=source, project_id=project_id,
                    embedding=pack(vec) if vec is not None else None,
                    created_at=db.now(), updated_at=db.now())
    return {"id": mid, "merged": False}


def keyword_search(con, query: str, k: int = 8) -> list[dict]:
    words = re.findall(r"\w{3,}", query.lower())[:8]
    if not words:
        return []
    where = " OR ".join("lower(text) LIKE ?" for _ in words)
    return db.rows(con, f"SELECT id, text, kind, source, project_id FROM memories WHERE {where} "
                        "ORDER BY updated_at DESC LIMIT ?", *[f"%{w}%" for w in words], k)


async def recall(con, llm, query: str, k: int = 8) -> list[dict]:
    vecs = await llm.embed([query])
    if vecs is None:
        return keyword_search(con, query, k)
    out = []
    for mid, score in top_k(vecs[0], _vectors(con, "memories"), k):
        if score >= RECALL_MIN:
            m = db.one(con, "SELECT id, text, kind, source, project_id FROM memories WHERE id=?", mid)
            m["score"] = round(score, 3)
            out.append(m)
    return out


def related(con, vec: np.ndarray, exclude_idea: int | None = None) -> list[dict]:
    items = [(("idea", i), b) for i, b in _vectors(con, "ideas") if i != exclude_idea]
    items += [(("project", i), b) for i, b in _vectors(con, "projects")]
    out = []
    for (kind, id_), score in top_k(vec, items, 3):
        if score < RELATED:
            continue
        col = "text" if kind == "idea" else "name"
        row = db.one(con, f"SELECT {col} AS text FROM {kind}s WHERE id=?", id_)
        out.append({"type": kind, "id": id_, "text": row["text"], "score": round(score, 3)})
    return out
