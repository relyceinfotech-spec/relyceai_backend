from app.chat import context as chat_context


def test_update_context_persists_summary(monkeypatch):
    saved = {}

    def _fake_persist(user_id: str, chat_id: str):
        saved["user_id"] = user_id
        saved["chat_id"] = chat_id
        saved["summary"] = chat_context.get_session_summary(user_id, chat_id)

    monkeypatch.setattr(chat_context, "persist_session_summary", _fake_persist)

    uid = "u_ctx_test"
    cid = "c_ctx_test"
    chat_context.clear_context(uid, cid)

    # Build enough turns to trigger rolling summary persistence.
    for i in range(chat_context.SUMMARY_TRIGGER_MESSAGES):
        chat_context.update_context_with_exchange(uid, cid, f"user {i}", f"assistant {i}")

    assert saved.get("user_id") == uid
    assert saved.get("chat_id") == cid
    assert isinstance(saved.get("summary"), str)
    assert len(saved.get("summary", "")) > 0
