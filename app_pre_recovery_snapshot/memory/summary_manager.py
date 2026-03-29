"""
Summary Orchestration Manager.
Handles determining when to summarize and applies the summary to the database 
while pruning the compressed messages.
"""
from typing import List, Dict, Any
from app.llm.token_counter import estimate_tokens
from app.memory.summarizer import generate_summary
from app.auth import get_firestore_db
from firebase_admin import firestore

# Summarize when the conversation context payload exceeds roughly 2000 tokens
SUMMARY_TRIGGER_TOKENS = 2000
# Keep the most recent N messages completely intact regardless of summary
KEEP_RECENT_COUNT = 6

def should_summarize(messages: List[Dict[str, str]], previous_summary: str = "") -> bool:
    """
    Determine if the current message buffer exceeds the token threshold.
    """
    # If the user has explicitly few messages, skip
    if len(messages) <= KEEP_RECENT_COUNT:
        return False
        
    tokens = estimate_tokens(messages, previous_summary)
    return tokens > SUMMARY_TRIGGER_TOKENS


async def summarize_if_needed(user_id: str, session_id: str, messages: List[Dict[str, str]], llm_client: Any) -> None:
    """
    Background worker function that checks the token count,
    generates a summary over the older context, updates the Firestore session, 
    and removes the compressed messages.
    """
    # 1. Fetch Session to get existing summary (incremental capability)
    db = get_firestore_db()
    if not db:
        return
        
    session_ref = db.collection('users').document(user_id).collection('chatSessions').document(session_id)
    session_doc = session_ref.get()
    previous_summary = ""
    if session_doc.exists:
        previous_summary = session_doc.to_dict().get("summary", "")

    if not should_summarize(messages, previous_summary):
        return

    # Extract the actual messages to compress (excluding the latest block)
    messages_to_compress = messages[:-KEEP_RECENT_COUNT]
    
    if not messages_to_compress:
        return

    print(f"[SummaryManager] Triggering summarization for {len(messages_to_compress)} messages (Session: {session_id})")
    
    # 2. Generate Incremental Summary
    new_summary = await generate_summary(llm_client, messages_to_compress, previous_summary)
    if not new_summary:
        return

    print(f"[SummaryManager] Generated summary ({len(new_summary)} chars). Applying to database...")
    
    try:
        session_ref.update({
            "summary": new_summary,
            "summary_tokens": estimate_tokens([], new_summary),
            "updatedAt": firestore.SERVER_TIMESTAMP
        })
    except Exception as e:
        print(f"[SummaryManager] Failed to update session summary: {e}")
        return

    # 3. Archive old messages (instead of deleting)
    messages_ref = session_ref.collection('messages')
    
    batch = db.batch()
    archive_count = 0
    
    for msg in messages_to_compress:
        msg_id = msg.get('id')
        if msg_id:
            msg_doc = messages_ref.document(msg_id)
            batch.update(msg_doc, {"archived": True})
            archive_count += 1
            
    if archive_count > 0:
        try:
            batch.commit()
            print(f"[SummaryManager] Automatically archived {archive_count} compressed messages in Firestore.")
        except Exception as e:
            print(f"[SummaryManager] Failed to commit archive batch: {e}")
