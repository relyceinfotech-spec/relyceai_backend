from __future__ import annotations

from app.capabilities.base import BaseCapabilityService


class BusinessCapabilityService(BaseCapabilityService):
    name = "business"
    default_mode = "business"
