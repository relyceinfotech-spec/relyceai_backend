import asyncio

from app.platform.task_queue import AgentTaskQueue
from app.platform.types import CapabilityRequest
import app.platform.task_queue as task_queue_mod
import app.chat.runtime_helpers as runtime_helpers


class _FakePlatform:
    async def run_stream(self, request):
        yield '[INFO] {"event":"task_progress","message":"Planning","percent":10}'
        yield '[INFO] {"event":"source","url":"https://example.com","title":"Example","type":"invalid"}'
        yield "Hello"
        yield " world"


async def _collect_until_done(queue: AgentTaskQueue, task_id: str, timeout_s: float = 5.0):
    deadline = asyncio.get_running_loop().time() + timeout_s
    last_seq = 0
    events = []
    while asyncio.get_running_loop().time() < deadline:
        new_events = await queue.get_events(task_id, after_seq=last_seq)
        if new_events:
            events.extend(new_events)
            last_seq = max(last_seq, max(int(e.get("seq", 0)) for e in new_events))
            if any(e.get("event") == "done" for e in new_events):
                return events
        await asyncio.sleep(0.05)
    raise AssertionError("Timed out waiting for done event")


def test_v2_success_order_and_required_fields(monkeypatch):
    async def _run():
        monkeypatch.setattr(task_queue_mod, "get_ai_platform", lambda: _FakePlatform())
        queue = AgentTaskQueue()
        await queue.start(fast_workers=1, heavy_workers=1)
        try:
            request = CapabilityRequest(user_query="hello", chat_mode="normal", user_id="u1", session_id="s1")
            task_id = await queue.submit(request)
            events = await _collect_until_done(queue, task_id)
        finally:
            await queue.shutdown()

        event_names = [e["event"] for e in events]
        assert event_names[0] == "progress"
        assert "message_start" in event_names
        assert event_names.count("final") == 1
        assert event_names.count("done") == 1
        assert event_names.index("final") < event_names.index("done")

        final_idx = event_names.index("final")
        assert "token" not in event_names[final_idx + 1 :]

        start_evt = next(e for e in events if e["event"] == "message_start")
        assert start_evt.get("stream_schema_version") == 2
        assert isinstance(start_evt.get("timestamp"), (int, float))

        token_events = [e for e in events if e["event"] == "token"]
        assert [e["token_seq"] for e in token_events] == sorted(e["token_seq"] for e in token_events)
        assert [e["token_seq"] for e in token_events] == list(range(1, len(token_events) + 1))
        assert [e["stream_seq"] for e in token_events] == [e["token_seq"] for e in token_events]

        source_evt = next(e for e in events if e["event"] == "source")
        assert source_evt["type"] == "web"
        assert source_evt["url"] == "https://example.com"

        final_evt = next(e for e in events if e["event"] == "final")
        assert isinstance(final_evt.get("answer"), str)
        assert isinstance(final_evt.get("metadata"), dict)

    asyncio.run(_run())


def test_heavy_lane_user_and_session_limits():
    async def _run():
        queue = AgentTaskQueue()
        await queue.start(fast_workers=1, heavy_workers=1, max_heavy_tasks_per_user=2, max_heavy_tasks_per_session=1)
        try:
            req1 = CapabilityRequest(user_query="research 1", chat_mode="deepsearch", user_id="u1", session_id="s1")
            await queue.submit(req1)

            # Session cap should block second heavy task in same session when cap=1.
            req_same_session = CapabilityRequest(
                user_query="research 2", chat_mode="deepsearch", user_id="u1", session_id="s1"
            )
            try:
                await queue.submit(req_same_session)
                raise AssertionError("Expected session limit error")
            except RuntimeError as e:
                assert "HEAVY_LANE_SESSION_LIMIT_REACHED" in str(e)

            # Different session still counts toward user cap.
            req2 = CapabilityRequest(user_query="research 3", chat_mode="deepsearch", user_id="u1", session_id="s2")
            await queue.submit(req2)
            req3 = CapabilityRequest(user_query="research 4", chat_mode="deepsearch", user_id="u1", session_id="s3")
            try:
                await queue.submit(req3)
                raise AssertionError("Expected user limit error")
            except RuntimeError as e:
                assert "HEAVY_LANE_LIMIT_REACHED" in str(e)
        finally:
            await queue.shutdown()

    asyncio.run(_run())


def test_streamed_tasks_persist_history_with_auth_uid(monkeypatch):
    async def _run():
        captured = {}

        def _fake_persist_exchange_and_schedule_summary(
            *,
            user_id,
            session_id,
            user_text,
            assistant_text,
            personality_id,
            chat_mode="normal",
            include_message_count=True,
        ):
            captured["user_id"] = user_id
            captured["session_id"] = session_id
            captured["user_text"] = user_text
            captured["assistant_text"] = assistant_text
            captured["personality_id"] = personality_id
            captured["chat_mode"] = chat_mode
            captured["include_message_count"] = include_message_count
            return "msg-123"

        monkeypatch.setattr(task_queue_mod, "get_ai_platform", lambda: _FakePlatform())
        monkeypatch.setattr(
            runtime_helpers,
            "persist_exchange_and_schedule_summary",
            _fake_persist_exchange_and_schedule_summary,
        )

        queue = AgentTaskQueue()
        await queue.start(fast_workers=1, heavy_workers=1)
        try:
            req = CapabilityRequest(
                user_query="hello persist",
                chat_mode="normal",
                user_id="memory-user-id",
                session_id="session-77",
                metadata={
                    "auth_uid": "firebase-uid-9",
                    "personality_id": "persona-1",
                },
            )
            task_id = await queue.submit(req)
            await _collect_until_done(queue, task_id)
            task = await queue.get_task(task_id)
        finally:
            await queue.shutdown()

        assert captured["user_id"] == "firebase-uid-9"
        assert captured["session_id"] == "session-77"
        assert captured["user_text"] == "hello persist"
        assert captured["personality_id"] == "persona-1"
        assert captured["chat_mode"] == "normal"
        assert captured["include_message_count"] is True
        assert task and task.result and task.result.get("message_id") == "msg-123"

    asyncio.run(_run())
