"""
Files Router
Handles file uploads and secure usage tracking.
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from typing import Optional
import shutil
import os
import time
from datetime import datetime
from app.auth import get_firestore_db, get_current_user
from firebase_admin import firestore

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    user_info: dict = Depends(get_current_user)
):
    """
    Upload a file and increment user usage securely.
    """
    try:
        user_id = user_info["uid"]
        
        # 2. Save File Locally (for RAG processing)
        # Generate safe filename
        timestamp = int(time.time())
        original_name = os.path.basename(file.filename)
        safe_filename = f"{user_id}_{timestamp}_{original_name}"
        file_path = os.path.join(UPLOAD_DIR, safe_filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        file_size_bytes = os.path.getsize(file_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        # 3. Securely Update Firestore Usage
        db = get_firestore_db()
        user_ref = db.collection("users").document(user_id)
        
        # We use strict field increments to avoid race conditions
        # Note: 'usage' map must exist. create_user_profile ensures this.
        # If not, we might need a set(..., merge=True) fallback or precondition check.
        
        try:
            user_ref.update({
                "usage.totalFilesUploaded": firestore.Increment(1),
                "usage.storageUsedMB": firestore.Increment(file_size_mb),
                "usage.lastActivity": firestore.SERVER_TIMESTAMP
            })
        except Exception as db_e:
            print(f"[Files] Error updating usage for {user_id}: {db_e}")
            # If doc doesn't exist or usage map missing (rare if flow is correct)
            # Try set merge
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

    except Exception as e:
        print(f"[Files] Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


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
