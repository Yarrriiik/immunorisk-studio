"""
Authentication Service - Handles user registration, login and password management
"""
import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

USERS_DB_PATH = Path("users_db.json")
# Simple admin reset code (should be changed/deployed via env in real system)
ADMIN_RESET_CODE = "IMMUNOADMIN2026"


def hash_password(password: str) -> str:
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()


def load_users() -> Dict[str, Dict[str, Any]]:
    """Load users database from JSON file"""
    if USERS_DB_PATH.exists():
        try:
            with open(USERS_DB_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_users(users: Dict[str, Dict[str, Any]]) -> None:
    """Save users database to JSON file"""
    with open(USERS_DB_PATH, 'w', encoding='utf-8') as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


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
    password_hash = hash_password(password)
    
    if user["password_hash"] != password_hash:
        return False, None, "Неверное имя пользователя или пароль"
    
    # Return user data without password hash
    user_data = {
        "username": user["username"],
        "full_name": user["full_name"],
        "specialization": user["specialization"],
        "stats": user["stats"],
        "history": user["history"]
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
        analysis_record["id"] = f"P-{len(users[username]['history']) + 1:03d}"
        analysis_record["date"] = datetime.now().strftime("%d.%m.%Y %H:%M")
        users[username]["history"].append(analysis_record)
        save_users(users)


def get_user_history(username: str) -> list[Dict[str, Any]]:
    """Get user's analysis history"""
    users = load_users()
    if username in users:
        return users[username]["history"]
    return []


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
    old_hash = hash_password(old_password)

    if user["password_hash"] != old_hash:
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
