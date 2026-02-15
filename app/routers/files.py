"""
Files Router
Handles file uploads and secure usage tracking.
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, Request
from typing import Optional
import os
import re
import time
from datetime import datetime, timezone
from app.auth import get_firestore_db, get_current_user
from firebase_admin import firestore
from app.config import (
    MAX_UPLOAD_BYTES,
    UPLOAD_ALLOWED_MIME_TYPES,
    UPLOAD_ALLOWED_EXTENSIONS,
    DEFAULT_STORAGE_QUOTA_MB,
)

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def _validate_upload_meta(file: UploadFile, content_length: Optional[str]) -> str:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    original_name = os.path.basename(file.filename)
    original_name = re.sub(r'[^\w\-. ]', '_', original_name)
    if len(original_name) > 200:
        original_name = original_name[:200]
    ext = os.path.splitext(original_name)[1].lower()

    if UPLOAD_ALLOWED_EXTENSIONS and ext not in UPLOAD_ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="File type not allowed")

    content_type = (file.content_type or "").lower()
    if UPLOAD_ALLOWED_MIME_TYPES and content_type not in UPLOAD_ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail="MIME type not allowed")

    if content_length:
        try:
            total = int(content_length)
            # Allow small multipart overhead
            if total > (MAX_UPLOAD_BYTES + 1024 * 1024):
                raise HTTPException(status_code=413, detail="File too large")
        except ValueError:
            pass

    return original_name

def _save_upload_limited(upload: UploadFile, file_path: str, max_bytes: int) -> int:
    size = 0
    with open(file_path, "wb") as buffer:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                raise HTTPException(status_code=413, detail="File too large")
            buffer.write(chunk)
    return size

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    user_info: dict = Depends(get_current_user),
    request: Request = None,
):
    """
    Upload a file and increment user usage securely.
    """
    try:
        user_id = user_info["uid"]
        content_length = None
        if request is not None:
            content_length = request.headers.get("content-length")

        original_name = _validate_upload_meta(file, content_length)

        # 2. Save File Locally (for RAG processing)
        # Generate safe filename
        timestamp = int(time.time())
        safe_filename = f"{user_id}_{timestamp}_{original_name}"
        file_path = os.path.join(UPLOAD_DIR, safe_filename)

        # Fetch user usage/quota (fail open if not found)
        db = get_firestore_db()
        user_ref = db.collection("users").document(user_id)
        user_doc = user_ref.get()
        usage = (user_doc.to_dict() or {}).get("usage", {}) if user_doc.exists else {}
        storage_used_mb = float(usage.get("storageUsedMB", 0) or 0)
        storage_quota_mb = float(usage.get("storageQuotaMB", DEFAULT_STORAGE_QUOTA_MB) or DEFAULT_STORAGE_QUOTA_MB)
        remaining_bytes = max(0, int((storage_quota_mb - storage_used_mb) * 1024 * 1024))

        if remaining_bytes <= 0:
            raise HTTPException(status_code=413, detail="Storage quota exceeded")

        max_bytes = min(MAX_UPLOAD_BYTES, remaining_bytes) if remaining_bytes else MAX_UPLOAD_BYTES

        try:
            file_size_bytes = _save_upload_limited(file, file_path, max_bytes)
        except Exception:
            if os.path.exists(file_path):
                os.remove(file_path)
            raise

        file_size_mb = file_size_bytes / (1024 * 1024)

        # 3. Securely Update Firestore Usage with atomic transaction
        @firestore.transactional
        def _check_and_update_usage(transaction, user_ref, file_size_mb, storage_quota_mb):
            doc = user_ref.get(transaction=transaction)
            if not doc.exists:
                return False
            data = doc.to_dict() or {}
            current_used = float((data.get("usage", {}) or {}).get("storageUsedMB", 0) or 0)
            if current_used + file_size_mb > storage_quota_mb:
                return False
            transaction.update(user_ref, {
                "usage.totalFilesUploaded": firestore.Increment(1),
                "usage.storageUsedMB": firestore.Increment(file_size_mb),
                "usage.lastActivity": firestore.SERVER_TIMESTAMP
            })
            return True
        
        try:
            success = _check_and_update_usage(db.transaction(), user_ref, file_size_mb, storage_quota_mb)
            if not success:
                if os.path.exists(file_path):
                    os.remove(file_path)
                raise HTTPException(status_code=413, detail="Storage quota exceeded")
        except HTTPException:
            raise
        except Exception as db_e:
            print(f"[Files] Error updating usage for {user_id}: {db_e}")
            user_ref.set({
                "usage": {
                    "totalFilesUploaded": firestore.Increment(1),
                    "storageUsedMB": firestore.Increment(file_size_mb),
                    "lastActivity": firestore.SERVER_TIMESTAMP
                }
            }, merge=True)

        return {
            "success": True,
            "filename": file.filename,
            "file_path": file_path,
            "size_mb": file_size_mb,
            "message": "File uploaded and usage updated"
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"[Files] Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    finally:
        try:
            await file.close()
        except Exception:
            pass


@router.delete("/delete/{user_id}/{filename}")
async def delete_file(
    user_id: str,
    filename: str,
    user_info: dict = Depends(get_current_user)
):
    """
    Delete an uploaded file for the authenticated user.
    Path user_id is ignored; token UID is enforced.
    """
    try:
        uid = user_info["uid"]
        safe_name = os.path.basename(filename)

        if safe_name != filename:
            raise HTTPException(status_code=400, detail="Invalid filename")

        if not safe_name.startswith(f"{uid}_"):
            raise HTTPException(status_code=403, detail="Cannot delete file")

        file_path = os.path.join(UPLOAD_DIR, safe_name)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        os.remove(file_path)

        return {"success": True, "message": "File deleted"}
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"[Files] Delete error: {e}")
        raise HTTPException(status_code=500, detail="Delete failed")
