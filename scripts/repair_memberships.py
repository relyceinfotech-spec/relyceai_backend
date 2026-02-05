import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone

# Ensure backend package root is on sys.path
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.auth import initialize_firebase, get_firestore_db


def infer_plan_from_membership(membership: dict) -> str:
    if not membership:
        return ""
    plan = membership.get("plan")
    if plan:
        return str(plan).lower()
    plan_name = membership.get("planName")
    if plan_name:
        return str(plan_name).strip().lower()
    payment_history = membership.get("paymentHistory")
    if isinstance(payment_history, dict) and payment_history:
        try:
            def get_date(item):
                data = item[1] or {}
                return data.get("date") or ""
            latest = max(payment_history.items(), key=get_date)
            plan = latest[1].get("plan")
            if plan:
                return str(plan).lower()
        except Exception:
            pass
    return ""


def normalize_role(role_value: str) -> str:
    if not role_value:
        return ""
    role = str(role_value).strip().lower()
    if role == "super_admin":
        return "superadmin"
    return role


def should_fix_user(data: dict) -> bool:
    role = data.get("role")
    membership = data.get("membership") or {}
    needs_role = not normalize_role(role)
    needs_plan = not membership.get("plan")
    return needs_role or needs_plan


def repair_users(apply: bool):
    initialize_firebase()
    db = get_firestore_db()
    users = db.collection("users").stream()

    fixed = 0
    reviewed = 0

    for user in users:
        reviewed += 1
        data = user.to_dict() or {}
        updates = {}

        role = data.get("role")
        normalized_role = normalize_role(role)
        if not normalized_role:
            updates["role"] = "user"
        elif normalized_role != role:
            updates["role"] = normalized_role

        membership = data.get("membership") or {}
        if membership and not membership.get("plan"):
            inferred = infer_plan_from_membership(membership)
            if inferred:
                updates["membership.plan"] = inferred
                updates["membership.planName"] = inferred.capitalize()

        if updates:
            fixed += 1
            if apply:
                user.reference.update(updates)
                db.collection("auditLogs").add({
                    "action": "REPAIR_PASS",
                    "by": "system",
                    "target": user.id,
                    "details": updates,
                    "timestamp": datetime.now(timezone.utc),
                    "time": datetime.now(timezone.utc).timestamp()
                })

    return reviewed, fixed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Apply changes (default is dry-run)")
    args = parser.parse_args()

    reviewed, fixed = repair_users(apply=args.apply)
    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"[{mode}] Reviewed: {reviewed}, Would Fix: {fixed}")


if __name__ == "__main__":
    main()
