from __future__ import annotations

import asyncio
import json
import time
import uuid
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.context.response_contract import build_final_answer_payload
from app.observability.metrics_collector import get_metrics_collector
from app.platform.mode_routing import select_queue_lane
from app.platform.service import get_ai_platform
from app.platform.types import CapabilityRequest
from app.chat.mode_mapper import normalize_chat_mode


@dataclass
class QueuedTask:
    task_id: str
    request: CapabilityRequest
    lane: str = "fast"
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    events: List[Dict[str, Any]] = field(default_factory=list)
    seq: int = 0
    cancel_requested: bool = False
    confirmation: Optional[bool] = None
    final_emitted: bool = False
    done_emitted: bool = False


class AgentTaskQueue:
    def __init__(self):
        self._fast_queue: asyncio.Queue[str] = asyncio.Queue()
        self._heavy_queue: asyncio.Queue[str] = asyncio.Queue()
        self._tasks: Dict[str, QueuedTask] = {}
        self._fast_workers: List[asyncio.Task] = []
        self._heavy_workers: List[asyncio.Task] = []
        self._active_runs: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        self._running = False
        self._fast_worker_count = 0
        self._heavy_worker_count = 0
        self._max_heavy_tasks_per_user = 2
        self._max_heavy_tasks_per_session = 0
        self._stream_schema_version = 2

    def _emit(self, task: QueuedTask, event: str, **payload: Any) -> None:
        # Deterministic stream guard:
        # after final, only done is allowed.
        if task.done_emitted:
            return
        if task.final_emitted and event != "done":
            return
        if event == "final":
            if task.final_emitted:
                return
            task.final_emitted = True
        if event == "done":
            if task.done_emitted:
                return
            task.done_emitted = True
        now = time.time()
        task.seq += 1
        task.events.append({"seq": task.seq, "event": event, "timestamp": now, **payload})

    def _choose_lane(self, request: CapabilityRequest) -> str:
        return select_queue_lane(chat_mode=str(request.chat_mode or "smart"), user_query=str(request.user_query or ""))

    def _count_heavy_tasks_for_user(self, user_id: str) -> int:
        if not user_id:
            return 0
        total = 0
        for t in self._tasks.values():
            if t.lane != "heavy":
                continue
            if str(getattr(t.request, "user_id", "")) != str(user_id):
                continue
            if t.status in {"queued", "running"}:
                total += 1
        return total

    def _count_heavy_tasks_for_session(self, session_id: str) -> int:
        if not session_id:
            return 0
        total = 0
        for t in self._tasks.values():
            if t.lane != "heavy":
                continue
            if str(getattr(t.request, "session_id", "")) != str(session_id):
                continue
            if t.status in {"queued", "running"}:
                total += 1
        return total

    def _ensure_final_payload_shape(self, payload: Dict[str, Any], request: CapabilityRequest) -> Dict[str, Any]:
        out = dict(payload or {})
        out["answer"] = str(out.get("answer") or "").strip()
        metadata = out.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        metadata.setdefault("mode", normalize_chat_mode(str(request.chat_mode or "smart")))
        metadata.setdefault("stream_schema_version", self._stream_schema_version)
        out["metadata"] = metadata
        return out

    def _emit_progress_from_info(self, task: QueuedTask, payload: Dict[str, Any], event_name: str) -> Optional[str]:
        if event_name == "final_answer":
            answer_text = str(payload.get("answer") or "").strip()
            return answer_text or None

        if event_name == "source":
            src_url = str(payload.get("url") or payload.get("link") or "").strip()
            if src_url:
                source_type = str(payload.get("type") or "web").strip().lower()
                if source_type not in {"web", "document", "database", "tool"}:
                    source_type = "web"
                self._emit(
                    task,
                    "source",
                    message_id=task.task_id,
                    id=str(payload.get("id") or uuid.uuid4().hex[:10]),
                    url=src_url,
                    title=str(payload.get("title") or payload.get("name") or "").strip() or None,
                    provider=str(payload.get("provider") or "").strip() or None,
                    confidence=payload.get("confidence"),
                    type=source_type,
                )
                return None

        if event_name in {"tool_call", "tool_result"}:
            result_count = payload.get("result_count")
            try:
                if result_count is not None:
                    result_count = int(result_count)
            except Exception:
                result_count = None

            result_items = payload.get("result_items")
            if not isinstance(result_items, list):
                result_items = None

            self._emit(
                task,
                event_name,
                message_id=task.task_id,
                call_id=str(payload.get("call_id") or payload.get("id") or uuid.uuid4().hex[:10]),
                tool=str(payload.get("tool") or payload.get("name") or "unknown"),
                args_preview=str(payload.get("args_preview") or payload.get("query") or "").strip() or None,
                status=str(payload.get("status") or ("started" if event_name == "tool_call" else "ok")),
                error=str(payload.get("error") or "").strip() or None,
                node_id=str(payload.get("node_id") or "").strip() or None,
                result_count=result_count,
                result_items=result_items,
                result_hint=str(payload.get("result_hint") or "").strip() or None,
                read_title=str(payload.get("read_title") or "").strip() or None,
                read_url=str(payload.get("read_url") or "").strip() or None,
            )
            return None

        state = str(payload.get("agent_state") or payload.get("state") or event_name or "progress")
        label = str(
            payload.get("label")
            or payload.get("topic")
            or payload.get("tool")
            or payload.get("message")
            or state
        )

        # Human-readable labels for planner milestones when the upstream event
        # only provides structural metadata (mode/node_count/similarity).
        planning_mode = str(payload.get("mode") or "").strip().lower()
        if state == "planning_complete":
            node_count = payload.get("node_count")
            if planning_mode == "query_cache_hit":
                label = "Plan ready - matched recent query cache"
            elif planning_mode == "retrieval_hit":
                label = "Plan ready - matched verified knowledge cache"
            elif planning_mode == "confidence_gate":
                label = "Plan ready - direct response path"
            elif planning_mode == "fast_path":
                label = "Plan ready - fast tool path"
            elif isinstance(node_count, int) and node_count > 0:
                label = f"Plan ready with {node_count} steps"
            else:
                label = "Plan ready"
        progress_payload: Dict[str, Any] = {
            "message_id": task.task_id,
            "state": state,
            "label": label,
        }
        if isinstance(payload.get("percent"), (int, float)):
            progress_payload["percent"] = float(payload["percent"])

        # Keep useful metadata for optional UI.
        for k in (
            "followups",
            "action_chips",
            "followup_mode",
            "confidence",
            "trust",
            "freshness",
            "mode",
            "query_type",
            "node_count",
            "similarity",
            "lane",
            "worker",
            "queue_wait_ms",
            "budget_ms",
            "failures",
            "reason",
            "tool",
            "status",
            "error",
        ):
            if k in payload:
                progress_payload[k] = payload[k]

        self._emit(task, "progress", **progress_payload)
        return None

    def _persist_completed_exchange(self, task: QueuedTask, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Persist queued chat exchanges so history survives page refresh.
        Uses `auth_uid` from metadata when available to avoid storing under memory-only ids.
        """
        out = dict(payload or {})
        try:
            session_id = str(getattr(task.request, "session_id", "") or "").strip()
            if not session_id:
                return out

            metadata = task.request.metadata if isinstance(task.request.metadata, dict) else {}
            auth_uid = str(metadata.get("auth_uid") or getattr(task.request, "user_id", "") or "").strip()
            if not auth_uid:
                return out

            user_query = str(getattr(task.request, "user_query", "") or "").strip()
            # Persist the full normalized response first so refresh/history
            # matches what users saw during streaming; fallback to answer.
            answer = str(out.get("response") or out.get("answer") or "").strip()
            if not user_query or not answer:
                return out

            personality_id = metadata.get("personality_id")
            from app.chat.runtime_helpers import persist_exchange_and_schedule_summary

            persisted_chat_mode = str(
                metadata.get("requested_mode_raw")
                or getattr(task.request, "chat_mode", "smart")
                or "smart"
            ).strip().lower() or "smart"

            message_id = persist_exchange_and_schedule_summary(
                user_id=auth_uid,
                session_id=session_id,
                user_text=user_query,
                assistant_text=answer,
                personality_id=personality_id,
                chat_mode=persisted_chat_mode,
                include_message_count=True,
            )
            if message_id:
                out["message_id"] = message_id
        except Exception as e:
            print(f"[TaskQueue] history_persist_failed type={e.__class__.__name__} msg={e}")
        return out

    async def _run_streaming_task(self, platform, task: QueuedTask) -> Dict[str, Any]:
        full_response = ""
        final_answer = ""
        confidence = 0.0
        final_info_meta: Dict[str, Any] = {}
        token_seq = 0
        self._emit(
            task,
            "message_start",
            message_id=task.task_id,
            mode=normalize_chat_mode(str(task.request.chat_mode or "smart")),
            stream_schema_version=self._stream_schema_version,
        )
        async for chunk in platform.run_stream(task.request):
            if chunk.strip().startswith("[INFO]"):
                raw = chunk.replace("[INFO]", "", 1).strip()
                try:
                    payload = json.loads(raw) if raw.startswith("{") else {"message": raw}
                except Exception:
                    payload = {"message": raw}
                event_name = str(payload.pop("event", "task_progress") or "task_progress")
                async with self._lock:
                    answer_text = self._emit_progress_from_info(task, payload, event_name)
                if event_name == "final_answer":
                    raw_confidence = payload.get("confidence")
                    if isinstance(raw_confidence, (int, float)):
                        confidence = float(raw_confidence)
                    snapshot = {
                        "critic_confidence": payload.get("critic_confidence"),
                        "repair_attempts": payload.get("repair_attempts"),
                        "unsupported_claims": payload.get("unsupported_claims"),
                        "evidence_quality": payload.get("evidence_quality"),
                        "tool_memory_hits": payload.get("tool_memory_hits"),
                        "reliability_budget_exhausted": payload.get("reliability_budget_exhausted"),
                        "fallback_used": payload.get("fallback_used"),
                    }
                    for k, v in snapshot.items():
                        if v is not None:
                            final_info_meta[k] = v
                    answer_text = answer_text or str(payload.get("answer") or "").strip()
                    if answer_text:
                        final_answer = answer_text
                continue
            if str(chunk).strip():
                text = str(chunk)
                full_response += text
                token_seq += 1
                async with self._lock:
                    self._emit(
                        task,
                        "token",
                        message_id=task.task_id,
                        token_seq=token_seq,  # legacy field for one release cycle
                        stream_seq=token_seq,  # canonical token sequence field
                        text=text,
                    )
        payload = build_final_answer_payload(
            {"response": full_response or final_answer, "confidence": confidence},
            user_query=task.request.user_query,
            chat_mode=task.request.chat_mode,
        )
        normalized = self._ensure_final_payload_shape(payload, task.request)
        if not str(normalized.get("response") or "").strip():
            normalized["response"] = str(full_response or final_answer or "").strip()
        if not str(normalized.get("answer") or "").strip():
            normalized["answer"] = str(final_answer or full_response or "").strip()
        metadata = normalized.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        for k, v in final_info_meta.items():
            if v is not None:
                metadata[k] = v
        normalized["metadata"] = metadata
        return normalized

    async def start(
        self,
        worker_count: Optional[int] = None,
        fast_workers: Optional[int] = None,
        heavy_workers: Optional[int] = None,
        max_heavy_tasks_per_user: Optional[int] = None,
        max_heavy_tasks_per_session: Optional[int] = None,
    ) -> None:
        if self._running:
            return
        self._running = True
        if fast_workers is None and heavy_workers is None and worker_count is not None:
            # Backward compatibility: split worker_count with fast-heavy bias.
            total = max(2, int(worker_count))
            fast_workers = max(1, total - 1)
            heavy_workers = 1
        self._fast_worker_count = max(1, int(fast_workers or 6))
        self._heavy_worker_count = max(1, int(heavy_workers or 2))
        self._max_heavy_tasks_per_user = max(1, int(max_heavy_tasks_per_user or 2))
        self._max_heavy_tasks_per_session = max(0, int(max_heavy_tasks_per_session or 0))

        for i in range(self._fast_worker_count):
            self._fast_workers.append(
                asyncio.create_task(self._worker_loop(i, lane="fast"), name=f"queue-fast-{i}")
            )
        for i in range(self._heavy_worker_count):
            self._heavy_workers.append(
                asyncio.create_task(self._worker_loop(i, lane="heavy"), name=f"queue-heavy-{i}")
            )

    async def shutdown(self) -> None:
        self._running = False
        for w in [*self._fast_workers, *self._heavy_workers]:
            w.cancel()
        if self._fast_workers or self._heavy_workers:
            await asyncio.gather(*self._fast_workers, *self._heavy_workers, return_exceptions=True)
        self._fast_workers.clear()
        self._heavy_workers.clear()

    async def submit(self, request: CapabilityRequest) -> str:
        metrics = get_metrics_collector()
        raw_mode = str(request.chat_mode or "smart").strip().lower() or "smart"
        lane = select_queue_lane(chat_mode=raw_mode, user_query=str(request.user_query or ""))
        request.chat_mode = normalize_chat_mode(raw_mode)
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        async with self._lock:
            if not isinstance(request.metadata, dict):
                request.metadata = {}
            request.metadata["requested_mode_raw"] = raw_mode
            request.metadata["lane"] = lane
            if lane == "heavy":
                owner = str(getattr(request, "user_id", "") or "")
                if owner and self._count_heavy_tasks_for_user(owner) >= self._max_heavy_tasks_per_user:
                    metrics.record_custom_counter("heavy_lane_limit_denied")
                    raise RuntimeError("HEAVY_LANE_LIMIT_REACHED")
                session_id = str(getattr(request, "session_id", "") or "")
                if (
                    self._max_heavy_tasks_per_session > 0
                    and session_id
                    and self._count_heavy_tasks_for_session(session_id) >= self._max_heavy_tasks_per_session
                ):
                    metrics.record_custom_counter("heavy_lane_session_limit_denied")
                    raise RuntimeError("HEAVY_LANE_SESSION_LIMIT_REACHED")
            task = QueuedTask(task_id=task_id, request=request)
            task.lane = lane
            self._emit(task, "progress", message_id=task_id, state="queued", label="Queued", lane=lane)
            self._tasks[task_id] = task
        if lane == "heavy":
            await self._heavy_queue.put(task_id)
        else:
            await self._fast_queue.put(task_id)
        return task_id

    async def queue_depth(self, lane: Optional[str] = None) -> int:
        if lane == "fast":
            return self._fast_queue.qsize()
        if lane == "heavy":
            return self._heavy_queue.qsize()
        return self._fast_queue.qsize() + self._heavy_queue.qsize()

    async def get_task(self, task_id: str) -> Optional[QueuedTask]:
        async with self._lock:
            return self._tasks.get(task_id)

    async def get_events(self, task_id: str, after_seq: int = 0) -> List[Dict[str, Any]]:
        async with self._lock:
            t = self._tasks.get(task_id)
            if not t:
                return []
            return [e for e in t.events if int(e.get("seq", 0)) > int(after_seq)]

    async def cancel_task(self, task_id: str) -> Optional[QueuedTask]:
        runner: Optional[asyncio.Task] = None
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None
            if task.status in {"completed", "failed", "cancelled"}:
                return task
            task.cancel_requested = True
            self._emit(task, "progress", message_id=task.task_id, state="cancel_requested", label="Cancel requested")
            if task.status == "queued":
                task.status = "cancelled"
                task.completed_at = time.time()
                task.error = "Task cancelled"
                self._emit(task, "error", message_id=task.task_id, code="cancelled", message=task.error)
                self._emit(task, "done", message_id=task.task_id)
                return task
            runner = self._active_runs.get(task_id)
        if runner:
            runner.cancel()
        async with self._lock:
            return self._tasks.get(task_id)

    async def confirm_task(self, task_id: str, confirm: bool) -> Optional[QueuedTask]:
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None
            task.confirmation = bool(confirm)
            if isinstance(task.request.metadata, dict):
                task.request.metadata["confirmation"] = bool(confirm)
            self._emit(
                task,
                "progress",
                message_id=task.task_id,
                state="confirmed" if confirm else "denied",
                label="Action confirmed" if confirm else "Action denied",
            )
            return task

    async def _worker_loop(self, worker_idx: int, lane: str) -> None:
        platform = get_ai_platform()
        metrics = get_metrics_collector()
        while self._running:
            queue = self._heavy_queue if lane == "heavy" else self._fast_queue
            task_id = await queue.get()
            try:
                async with self._lock:
                    task = self._tasks.get(task_id)
                    if not task:
                        continue
                    if task.cancel_requested:
                        task.status = "cancelled"
                        task.completed_at = time.time()
                        self._emit(task, "error", message_id=task.task_id, code="cancelled", message="Task cancelled")
                        self._emit(task, "done", message_id=task.task_id)
                        continue
                    task.status = "running"
                    task.started_at = time.time()
                    queue_wait_ms = round((task.started_at - task.created_at) * 1000, 2)
                    if not isinstance(task.request.metadata, dict):
                        task.request.metadata = {}
                    task.request.metadata["queue_wait_ms"] = queue_wait_ms
                    metrics.record_custom_counter(f"queue_wait_lane_{lane}_samples")
                    metrics.record_custom_counter(f"queue_wait_lane_{lane}_ms_total", int(max(0.0, queue_wait_ms)))
                    self._emit(
                        task,
                        "progress",
                        message_id=task.task_id,
                        state="running",
                        label=f"Running on {lane} lane",
                        lane=lane,
                        worker=worker_idx,
                        queue_wait_ms=queue_wait_ms,
                    )

                try:
                    run_task = asyncio.create_task(self._run_streaming_task(platform, task), name=f"agent-run-{task_id}")
                    async with self._lock:
                        self._active_runs[task_id] = run_task
                    payload = await run_task
                    payload = self._persist_completed_exchange(task, payload)
                    async with self._lock:
                        task.status = "completed"
                        task.completed_at = time.time()
                        task.result = payload
                        self._emit(
                            task,
                            "progress",
                            message_id=task.task_id,
                            state="completed",
                            label="Completed",
                            latency_ms=round((task.completed_at - (task.started_at or task.created_at)) * 1000, 2),
                        )
                        self._emit(
                            task,
                            "final",
                            message_id=task.task_id,
                            answer=str(payload.get("answer") or ""),
                            summary=payload.get("summary"),
                            metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                        )
                        self._emit(task, "done", message_id=task.task_id)
                except asyncio.CancelledError:
                    async with self._lock:
                        task.status = "cancelled"
                        task.completed_at = time.time()
                        task.error = "Task cancelled"
                        self._emit(task, "error", message_id=task.task_id, code="cancelled", message=task.error)
                        self._emit(task, "done", message_id=task.task_id)
                except Exception as e:
                    print(f"[TaskQueue] task_failed id={task_id} type={e.__class__.__name__} msg={str(e)}")
                    traceback.print_exc()
                    async with self._lock:
                        task.status = "failed"
                        task.completed_at = time.time()
                        task.error = str(e)
                        self._emit(
                            task,
                            "error",
                            message_id=task.task_id,
                            code=e.__class__.__name__,
                            message=str(e)[:500],
                        )
                        self._emit(task, "done", message_id=task.task_id)
            finally:
                async with self._lock:
                    self._active_runs.pop(task_id, None)
                queue.task_done()


_queue_singleton: AgentTaskQueue | None = None


def get_task_queue() -> AgentTaskQueue:
    global _queue_singleton
    if _queue_singleton is None:
        _queue_singleton = AgentTaskQueue()
    return _queue_singleton
