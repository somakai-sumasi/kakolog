"""保存のビジネスロジック。"""

from pathlib import Path

from ..chunker import TurnChunk, merge_short_turns
from ..cleaner import clean_text
from ..config import SIMILARITY_THRESHOLD, is_excluded, is_excluded_session
from ..db import transaction
from ..embedder import cosine_similarity, embed_documents
from ..extractor import extract_conversations, read_session_meta
from ..models import ConversationPair, Memory, SearchScope
from ..repository import (
    MemoryToSave,
    fetch_embeddings_by_ids,
    fetch_memories_by_ids,
    find_memory_by_content,
    insert_memory,
    search_vec,
)
from ..transcript import parse_jsonl
from . import touch_memories


def _build_chunks(transcript_path: str | Path) -> list[TurnChunk]:
    """JSONL → 会話抽出 → クリーニング → チャンク分割。"""
    messages = parse_jsonl(transcript_path)
    pairs = extract_conversations(messages)
    cleaned = [
        ConversationPair(
            user_turn=clean_text(p.user_turn),
            agent_turn=clean_text(p.agent_turn),
            timestamp=p.timestamp,
        )
        for p in pairs
    ]
    return merge_short_turns(cleaned)


def save_session(
    session_id: str, transcript_path: str, project_path: str | None = None
) -> int:
    meta = read_session_meta(transcript_path)
    if is_excluded_session(transcript_path, meta.entrypoint):
        return 0
    resolved_project_path = project_path if project_path is not None else meta.cwd
    if is_excluded(resolved_project_path):
        return 0
    chunks = _build_chunks(transcript_path)
    if not chunks:
        return 0

    texts = [c.content for c in chunks]
    embeddings = embed_documents(texts)

    with transaction():
        count = 0
        for chunk, emb in zip(chunks, embeddings):
            ts = chunk.timestamp or meta.first_timestamp
            existing = find_memory_by_content(chunk.content, resolved_project_path)
            if existing:
                continue
            similar = _find_similar(emb, resolved_project_path)
            if similar:
                touch_memories([similar])
                continue
            insert_memory(
                MemoryToSave(
                    session_id=session_id,
                    user_turn=chunk.user_turn,
                    agent_turn=chunk.agent_turn,
                    content=chunk.content,
                    embedding=emb,
                    project_path=resolved_project_path,
                    last_accessed_at=ts,
                ),
            )
            count += 1

    return count


def _find_similar(
    embedding: list[float],
    project_path: str | None,
) -> Memory | None:
    """同一project_path内で類似度がSIMILARITY_THRESHOLD以上のメモリを返す。"""
    candidate_ids = search_vec(embedding, limit=5)
    if not candidate_ids:
        return None
    scope = SearchScope.of(project_path=project_path)
    memories = fetch_memories_by_ids(candidate_ids, scope)
    if not memories:
        return None
    emb_map = fetch_embeddings_by_ids([m.id for m in memories])
    for m in memories:
        if m.id not in emb_map:
            continue
        if cosine_similarity(embedding, emb_map[m.id]) >= SIMILARITY_THRESHOLD:
            return m
    return None
