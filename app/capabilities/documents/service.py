from __future__ import annotations

from app.capabilities.base import BaseCapabilityService


class DocumentsCapabilityService(BaseCapabilityService):
    name = "documents"
    default_mode = "documents"
