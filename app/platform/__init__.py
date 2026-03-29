from .types import CapabilityRequest, CapabilityResponse

__all__ = [
    "AIPlatformService",
    "get_ai_platform",
    "CapabilityRequest",
    "CapabilityResponse",
    "AgentTaskQueue",
    "get_task_queue",
]


def __getattr__(name):
    if name in {"AIPlatformService", "get_ai_platform"}:
        from .service import AIPlatformService, get_ai_platform

        return {
            "AIPlatformService": AIPlatformService,
            "get_ai_platform": get_ai_platform,
        }[name]
    if name in {"AgentTaskQueue", "get_task_queue"}:
        from .task_queue import AgentTaskQueue, get_task_queue

        return {
            "AgentTaskQueue": AgentTaskQueue,
            "get_task_queue": get_task_queue,
        }[name]
    raise AttributeError(f"module 'app.platform' has no attribute {name!r}")
