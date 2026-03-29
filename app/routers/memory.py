from __future__ import annotations

from datetime import datetime
from fastapi import APIRouter, Request

from app.llm.prompts import MEMORY_IMPORT_PARSE_PROMPT, MEMORY_SUMMARY_SYSTEM_PROMPT
from app.config import FAST_MODEL

router = APIRouter(tags=["memory"])


@router.get("/api/memories/{user_id}")
async def get_memories(user_id: str):
    """Get all auto-detected memories for a user."""
    from app.chat.smart_memory import get_all_memories_for_api

    memories = get_all_memories_for_api(user_id)
    return {"memories": memories, "count": len(memories)}


@router.delete("/api/memories/{user_id}/{memory_id}")
async def delete_memory_endpoint(user_id: str, memory_id: str):
    """Delete a specific memory."""
    from app.chat.smart_memory import delete_memory

    success = delete_memory(user_id, memory_id)
    if success:
        return {"status": "deleted", "memory_id": memory_id}
    return {"status": "error", "message": "Failed to delete memory"}


@router.post("/api/memories/import")
async def import_memories_endpoint(request: Request):
    """Import memories from pasted exports using LLM parsing."""
    body = await request.json()
    user_id = body.get("user_id", "")
    raw_text = body.get("text", "")

    if not user_id or not raw_text:
        return {"status": "error", "message": "user_id and text required"}

    if len(raw_text) > 10000:
        return {"status": "error", "message": "Text too long (max 10000 chars)"}

    try:
        from app.llm.router import get_openrouter_client
        from app.chat.smart_memory import MemoryEntry, store_memory
        import json as json_lib

        client = get_openrouter_client()
        response = await client.chat.completions.create(
            model=FAST_MODEL,
            messages=[
                {"role": "system", "content": MEMORY_IMPORT_PARSE_PROMPT},
                {"role": "user", "content": f"Parse these memories:\n\n{raw_text[:8000]}"},
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=2000,
            temperature=0.1,
        )

        result_text = response.choices[0].message.content
        parsed = json_lib.loads(result_text)
        entries_data = parsed if isinstance(parsed, list) else parsed.get("memories", parsed.get("entries", []))

        stored_count = 0
        stored_entries = []
        for item in entries_data:
            if not isinstance(item, dict) or not item.get("content"):
                continue
            entry = MemoryEntry(
                content=item["content"][:200],
                category=item.get("category", "context"),
                importance=min(1.0, max(0.1, float(item.get("importance", 0.6)))),
                source="imported",
            )
            if store_memory(user_id, entry):
                stored_count += 1
                stored_entries.append({"content": entry.content, "category": entry.category})

        return {"status": "success", "imported": stored_count, "entries": stored_entries}
    except Exception as e:
        print(f"[MemoryImport] Error: {e}")
        return {"status": "error", "message": f"Import failed: {str(e)}"}


@router.get("/api/memories/export/{user_id}")
async def export_memories_endpoint(user_id: str):
    """Export all memories in a structured format for copying."""
    from app.chat.smart_memory import load_all_memories

    memories = load_all_memories(user_id, force_refresh=True)
    if not memories:
        return {"status": "empty", "export": "No memories stored yet."}

    groups = {}
    for mem in memories:
        cat = mem.category.title()
        if cat not in groups:
            groups[cat] = []
        date = mem.created_at[:10] if mem.created_at else "unknown"
        groups[cat].append(f"[{date}] - {mem.content}")

    export_lines = []
    category_order = ["Identity", "Profession", "Project", "Preference", "Context"]
    for cat in category_order:
        if cat in groups:
            export_lines.append(f"## {cat}")
            for line in groups[cat]:
                export_lines.append(line)
            export_lines.append("")

    return {"status": "success", "export": "\n".join(export_lines), "count": len(memories)}


@router.get("/api/memories/summary/{user_id}")
async def get_memory_summary(user_id: str):
    """Generate an LLM-powered prose summary of what Relyce knows about the user."""
    from app.chat.smart_memory import load_all_memories

    memories = load_all_memories(user_id, force_refresh=True)
    if not memories:
        return {"status": "empty", "summary": "No memories stored yet. Start chatting and Relyce will learn about you!"}

    try:
        from app.llm.router import get_openrouter_client

        mem_lines = []
        for mem in memories:
            mem_lines.append(f"[{mem.category}] {mem.content}")
        mem_text = "\n".join(mem_lines)

        client = get_openrouter_client()
        response = await client.chat.completions.create(
            model=FAST_MODEL,
            messages=[
                {"role": "system", "content": MEMORY_SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": f"Summarize these memories:\n\n{mem_text}"},
            ],
            max_completion_tokens=500,
            temperature=0.3,
        )

        summary = response.choices[0].message.content
        return {"status": "success", "summary": summary, "memory_count": len(memories)}
    except Exception as e:
        print(f"[MemorySummary] Error: {e}")
        fallback = "\n".join([f"- {m.content}" for m in memories[:20]])
        return {"status": "success", "summary": fallback, "memory_count": len(memories)}


@router.patch("/api/memories/{user_id}/{memory_id}")
async def update_memory_endpoint(user_id: str, memory_id: str, request: Request):
    """Update a memory's content."""
    body = await request.json()
    new_content = body.get("content", "")
    if not new_content:
        return {"status": "error", "message": "content required"}

    from app.chat.smart_memory import _get_entries_ref, _invalidate_cache

    try:
        ref = _get_entries_ref(user_id)
        if ref:
            ref.document(memory_id).update(
                {
                    "content": new_content[:200],
                    "last_used": datetime.utcnow().isoformat(),
                }
            )
            _invalidate_cache(user_id)
            return {"status": "updated", "memory_id": memory_id}
    except Exception as e:
        print(f"[MemoryUpdate] Error: {e}")
    return {"status": "error", "message": "Failed to update memory"}


@router.delete("/api/memories/clear/{user_id}")
async def clear_all_memories(user_id: str):
    """Delete all memories for a user."""
    from app.chat.smart_memory import load_all_memories, _get_entries_ref, _invalidate_cache

    try:
        memories = load_all_memories(user_id, force_refresh=True)
        ref = _get_entries_ref(user_id)
        if ref:
            for mem in memories:
                if mem.doc_id:
                    ref.document(mem.doc_id).delete()
            _invalidate_cache(user_id)
        return {"status": "cleared", "count": len(memories)}
    except Exception as e:
        print(f"[MemoryClear] Error: {e}")
        return {"status": "error", "message": str(e)}





