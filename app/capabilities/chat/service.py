from __future__ import annotations

from app.capabilities.base import BaseCapabilityService


class ChatCapabilityService(BaseCapabilityService):
    name = "chat"
    default_mode = "smart"
