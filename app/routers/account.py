from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, EmailStr

from app.auth import get_current_user
from app.models import Personality
from app.governance.login_rate_limiter import (
    check_rate_limit as check_login_rate_limit,
    record_failed_attempt,
    clear_attempts,
)
from app.chat.personalities import (
    get_all_personalities,
    create_custom_personality,
    get_personality_by_id,
)

router = APIRouter(tags=["account"])


class RateLimitRequest(BaseModel):
    email: EmailStr


@router.post("/auth/check-limit")
async def check_login_limit(request: RateLimitRequest, req: Request):
    ip = req.client.host if req.client else "unknown"
    return check_login_rate_limit(request.email, ip)


@router.post("/auth/record-failure")
async def record_login_failure(request: RateLimitRequest, req: Request):
    ip = req.client.host if req.client else "unknown"
    return record_failed_attempt(request.email, ip)


@router.post("/auth/clear-attempts")
async def clear_login_attempts(request: RateLimitRequest, req: Request):
    ip = req.client.host if req.client else "unknown"
    success = clear_attempts(request.email, ip)
    return {"success": success}


@router.get("/personalities/{user_id}")
async def get_personalities(user_id: str, user_info: dict = Depends(get_current_user)):
    uid = user_info["uid"]
    return {"success": True, "personalities": get_all_personalities(uid)}


@router.post("/personalities")
async def create_personality(request: Personality, user_info: dict = Depends(get_current_user)):
    user_id = user_info["uid"]
    result = create_custom_personality(
        user_id,
        request.name,
        request.description,
        request.prompt,
        request.content_mode,
        request.specialty,
    )
    if result:
        return {"success": True, "personality": result}
    raise HTTPException(status_code=500, detail="Failed to create personality")


@router.put("/personalities/{personality_id}")
async def update_personality(personality_id: str, request: Personality, user_info: dict = Depends(get_current_user)):
    from app.chat.personalities import update_custom_personality

    user_id = user_info["uid"]
    success = update_custom_personality(
        user_id,
        personality_id,
        request.name,
        request.description,
        request.prompt,
        request.content_mode,
        request.specialty,
    )

    if success:
        return {"success": True}
    raise HTTPException(status_code=404, detail="Personality not found or failed to update")


@router.delete("/personalities/{personality_id}")
async def delete_personality(personality_id: str, user_info: dict = Depends(get_current_user)):
    from app.chat.personalities import delete_custom_personality

    user_id = user_info["uid"]
    success = delete_custom_personality(user_id, personality_id)

    if success:
        return {"success": True}
    raise HTTPException(status_code=404, detail="Personality not found or failed to delete")

