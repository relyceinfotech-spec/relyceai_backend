import asyncio

from app.platform.task_queue import AgentTaskQueue
from app.platform.types import CapabilityRequest
import app.platform.task_queue as task_queue_mod


class _LegacyInfoPlatform:
    async def run_stream(self, request):
        yield '[INFO] {"event":"task_progress","message":"Planning","percent":10}'
        yield '[INFO] {"event":"final_answer","answer":"legacy final answer","confidence":0.81}'
        yield " legacy tail token"


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


def test_legacy_final_answer_translation_is_preserved(monkeypatch):
    async def _run():
        monkeypatch.setattr(task_queue_mod, "get_ai_platform", lambda: _LegacyInfoPlatform())
        queue = AgentTaskQueue()
        await queue.start(fast_workers=1, heavy_workers=1)
        try:
            request = CapabilityRequest(user_query="legacy", chat_mode="normal", user_id="u1", session_id="s1")
            task_id = await queue.submit(request)
            events = await _collect_until_done(queue, task_id)
        finally:
            await queue.shutdown()

        names = [e["event"] for e in events]
        assert names.count("final") == 1
        assert names.count("done") == 1
        assert names.index("final") < names.index("done")
        final_evt = next(e for e in events if e["event"] == "final")
        assert final_evt.get("answer") == "legacy final answer"
        assert isinstance(final_evt.get("metadata"), dict)

    asyncio.run(_run())
