"""
Authentication Service - Handles user registration, login and password management
"""
import hashlib
import hmac
import json
import os
import secrets
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

USERS_DB_PATH = Path(os.getenv("IMMUNORISK_USERS_DB", "users_db.local.json"))
ADMIN_RESET_CODE = os.getenv("IMMUNORISK_ADMIN_RESET_CODE")
PBKDF2_PREFIX = "pbkdf2_sha256"
PBKDF2_ITERATIONS = 390000


def hash_password(password: str) -> str:
    """Hash password using PBKDF2-HMAC-SHA256 with a random salt."""
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        PBKDF2_ITERATIONS,
    )
    return f"{PBKDF2_PREFIX}${PBKDF2_ITERATIONS}${salt}${digest.hex()}"


def verify_password(password: str, password_hash: str) -> bool:
    """Verify both new PBKDF2 hashes and legacy SHA256 hashes."""
    if password_hash.startswith(f"{PBKDF2_PREFIX}$"):
        try:
            _, iterations_str, salt, digest_hex = password_hash.split("$", 3)
            digest = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode("utf-8"),
                salt.encode("utf-8"),
                int(iterations_str),
            )
            return hmac.compare_digest(digest.hex(), digest_hex)
        except (ValueError, TypeError):
            return False

    legacy_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()
    return hmac.compare_digest(legacy_hash, password_hash)


def needs_rehash(password_hash: str) -> bool:
    return not password_hash.startswith(f"{PBKDF2_PREFIX}$")


def load_users() -> Dict[str, Dict[str, Any]]:
    """Load users database from JSON file"""
    if USERS_DB_PATH.exists():
        try:
            with open(USERS_DB_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_users(users: Dict[str, Dict[str, Any]]) -> None:
    """Save users database to JSON file"""
    USERS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(USERS_DB_PATH, 'w', encoding='utf-8') as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def _normalize_history_record(record: Dict[str, Any], index: int) -> tuple[Dict[str, Any], bool]:
    normalized = dict(record)
    changed = False

    analysis_id = normalized.get("analysis_id")
    legacy_id = normalized.get("id")
    if not analysis_id:
        if isinstance(legacy_id, str) and legacy_id.startswith("A-"):
            analysis_id = legacy_id
        else:
            analysis_id = f"A-{index:03d}"
        normalized["analysis_id"] = analysis_id
        changed = True

    if normalized.get("id") != analysis_id:
        normalized["id"] = analysis_id
        changed = True

    patient_id = normalized.get("patient_id")
    if not patient_id:
        if isinstance(legacy_id, str) and legacy_id.startswith("P-"):
            patient_id = legacy_id
        else:
            patient_id = "P-UNKNOWN"
        normalized["patient_id"] = patient_id
        changed = True
    elif not isinstance(patient_id, str):
        normalized["patient_id"] = str(patient_id)
        changed = True

    return normalized, changed


def _normalize_user_history(users: Dict[str, Dict[str, Any]], username: str) -> list[Dict[str, Any]]:
    history = users[username].get("history", [])
    normalized_history = []
    changed = False

    for index, record in enumerate(history, start=1):
        normalized_record, record_changed = _normalize_history_record(record, index)
        normalized_history.append(normalized_record)
        changed = changed or record_changed

    if changed:
        users[username]["history"] = normalized_history
        save_users(users)

    return normalized_history


def register_user(username: str, password: str, full_name: str, specialization: str) -> tuple[bool, str]:
    """
    Register a new user
    
    Returns:
        (success: bool, message: str)
    """
    users = load_users()
    
    if username in users:
        return False, "Пользователь с таким именем уже существует"
    
    if len(password) < 6:
        return False, "Пароль должен содержать минимум 6 символов"
    
    users[username] = {
        "username": username,
        "password_hash": hash_password(password),
        "full_name": full_name,
        "specialization": specialization,
        "created_at": datetime.now().isoformat(),
        "stats": {
            "total_patients": 0,
            "total_analyses": 0
        },
        "history": []
    }
    
    save_users(users)
    return True, "Регистрация успешна!"


def login_user(username: str, password: str) -> tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Login user
    
    Returns:
        (success: bool, user_data: Optional[Dict], message: str)
    """
    users = load_users()
    
    if username not in users:
        return False, None, "Неверное имя пользователя или пароль"
    
    user = users[username]
    password_hash = user.get("password_hash", "")

    if not verify_password(password, password_hash):
        return False, None, "Неверное имя пользователя или пароль"

    if needs_rehash(password_hash):
        user["password_hash"] = hash_password(password)
        users[username] = user
        save_users(users)
    
    # Return user data without password hash
    user_data = {
        "username": user["username"],
        "full_name": user["full_name"],
        "specialization": user["specialization"],
        "stats": user["stats"],
        "history": _normalize_user_history(users, username)
    }
    
    return True, user_data, "Вход выполнен успешно!"


def update_user_stats(username: str, increment_patients: int = 0, increment_analyses: int = 0) -> None:
    """Update user statistics"""
    users = load_users()
    if username in users:
        users[username]["stats"]["total_patients"] += increment_patients
        users[username]["stats"]["total_analyses"] += increment_analyses
        save_users(users)


def add_to_user_history(username: str, analysis_record: Dict[str, Any]) -> None:
    """Add analysis record to user history"""
    users = load_users()
    if username in users:
        analysis_id = f"A-{len(users[username]['history']) + 1:03d}"
        analysis_record["analysis_id"] = analysis_id
        # Keep legacy key for backward compatibility with old exports/history readers.
        analysis_record["id"] = analysis_id
        analysis_record["patient_id"] = str(analysis_record.get("patient_id") or "P-UNKNOWN")
        analysis_record["date"] = datetime.now().strftime("%d.%m.%Y %H:%M")
        users[username]["history"].append(analysis_record)
        save_users(users)


def get_user_history(username: str) -> list[Dict[str, Any]]:
    """Get user's analysis history"""
    users = load_users()
    if username in users:
        return _normalize_user_history(users, username)
    return []


def clear_user_history(username: str) -> tuple[bool, str]:
    """Clear user's history and reset derived counters."""
    users = load_users()
    if username not in users:
        return False, "Пользователь не найден"

    users[username]["history"] = []
    users[username]["stats"] = {
        "total_patients": 0,
        "total_analyses": 0,
    }
    save_users(users)
    return True, "История анализов очищена"


def get_user_stats(username: str) -> Dict[str, int]:
    """Get user statistics"""
    users = load_users()
    if username in users:
        return users[username]["stats"]
    return {"total_patients": 0, "total_analyses": 0}


def change_password(username: str, old_password: str, new_password: str) -> tuple[bool, str]:
    """
    Change user password (requires old password).

    Returns:
        (success: bool, message: str)
    """
    users = load_users()

    if username not in users:
        return False, "Пользователь не найден"

    if len(new_password) < 6:
        return False, "Новый пароль должен содержать минимум 6 символов"

    user = users[username]

    if not verify_password(old_password, user.get("password_hash", "")):
        return False, "Текущий пароль введен неверно"

    if old_password == new_password:
        return False, "Новый пароль не должен совпадать с текущим"

    user["password_hash"] = hash_password(new_password)
    user["password_changed_at"] = datetime.now().isoformat()
    users[username] = user
    save_users(users)

    return True, "Пароль успешно изменен"


def admin_reset_password(admin_code: str, username: str, new_password: str) -> tuple[bool, str]:
    """
    Reset user password using admin/secret code (no old password required).

    Returns:
        (success: bool, message: str)
    """
    if not ADMIN_RESET_CODE:
        return False, "Сброс пароля отключен: задайте IMMUNORISK_ADMIN_RESET_CODE"

    if admin_code != ADMIN_RESET_CODE:
        return False, "Неверный секретный код администратора"

    users = load_users()

    if username not in users:
        return False, "Пользователь не найден"

    if len(new_password) < 6:
        return False, "Новый пароль должен содержать минимум 6 символов"

    user = users[username]
    user["password_hash"] = hash_password(new_password)
    user["password_reset_by_admin_at"] = datetime.now().isoformat()
    users[username] = user
    save_users(users)

    return True, "Пароль пользователя успешно сброшен администратором"
