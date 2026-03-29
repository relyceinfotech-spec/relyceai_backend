from __future__ import annotations

from app.capabilities.base import BaseCapabilityService


class AutomationCapabilityService(BaseCapabilityService):
    name = "automation"
    default_mode = "agent"
