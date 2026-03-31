import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import traceback
from pathlib import Path

from ml_service import (
    predict_patient,
    get_model_info,
    get_available_cohorts,
    get_model_cohort_name,
    get_input_coverage,
)
from auth_service import (
    register_user,
    login_user,
    update_user_stats,
    add_to_user_history,
    get_user_history,
    get_user_stats,
    change_password,
    admin_reset_password,
    clear_user_history,
)
from form_generator import generate_dynamic_form, get_minimal_profile_status, get_input_examples
from report_generator import generate_pdf_report, generate_csv_history, REPORTLAB_AVAILABLE

# ===================== КОНФИГУРАЦИЯ СТРАНИЦЫ =====================
st.set_page_config(
    page_title="Immunorisk Studio",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== CSS СТИЛИ МЕДИЦИНСКОЙ ТЕМЫ =====================
st.markdown("""
<style>
    /* Основные цвета медицинской темы - синяя палитра */
    :root {
        --primary: #2a5c8a;
        --primary-light: #4a8bc5;
        --secondary: #3a9e7a;
        --secondary-light: #5bc49f;
        --accent: #e63946;
        --light: #f8f9fa;
        --light-blue: #e8f4fd;
        --medium-gray: #adb5bd;
        --dark: #343a40;
    }

    /* Основные стили */
    .main-header {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(42, 92, 138, 0.2);
    }

    .user-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin-bottom: 1.5rem;
        text-align: center;
        color: var(--dark);
    }

    .section-title {
        color: var(--primary);
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--secondary-light);
    }

    .subsection-title {
        color: var(--primary);
        font-size: 1.4rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1.2rem;
    }

    .small-title {
        color: var(--primary);
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }

    /* Метрики в красивых карточках */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid var(--secondary);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        height: 100%;
        transition: transform 0.3s ease;
        color: var(--dark);
    }

    .metric-card:hover {
        transform: translateY(-3px);
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--primary);
        margin: 0.5rem 0;
    }

    .metric-label {
        color: #4f5b67;
        font-size: 0.95rem;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }

    .metric-delta {
        font-size: 0.9rem;
        font-weight: 600;
    }

    .delta-positive {
        color: var(--secondary);
    }

    .delta-negative {
        color: var(--accent);
    }

    /* Плитки факторов риска */
    .factor-tile {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 2px solid #e9ecef;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        transition: all 0.2s ease;
        color: var(--dark);
    }

    .factor-tile:hover {
        border-color: var(--primary-light);
        box-shadow: 0 4px 12px rgba(74, 139, 197, 0.1);
    }

    .factor-tile.high {
        border-left: 4px solid var(--accent);
    }

    .factor-tile.medium {
        border-left: 4px solid #ffc107;
    }

    .factor-tile.low {
        border-left: 4px solid var(--secondary);
    }

    .factor-name {
        font-weight: 600;
        color: var(--dark);
        margin-bottom: 0.3rem;
        font-size: 1rem;
    }

    .factor-value {
        color: #4f5b67;
        font-size: 0.9rem;
        margin-bottom: 0.3rem;
    }

    .factor-impact {
        font-weight: 700;
        margin-top: 0.5rem;
        font-size: 0.85rem;
    }

    .impact-high { color: var(--accent); }
    .impact-medium { color: #d39e00; }
    .impact-low { color: var(--secondary); }

    /* Бейджи рисков */
    .risk-badge {
        padding: 0.6rem 1.5rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1rem;
        display: inline-block;
        text-align: center;
    }

    .risk-high {
        background: linear-gradient(135deg, rgba(230, 57, 70, 0.12) 0%, rgba(230, 57, 70, 0.06) 100%);
        color: var(--accent);
        border: 2px solid var(--accent);
    }

    .risk-medium {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.12) 0%, rgba(255, 193, 7, 0.06) 100%);
        color: #d39e00;
        border: 2px solid #ffc107;
    }

    .risk-low {
        background: linear-gradient(135deg, rgba(58, 158, 122, 0.12) 0%, rgba(58, 158, 122, 0.06) 100%);
        color: var(--secondary);
        border: 2px solid var(--secondary);
    }

    /* Карточки для рекомендаций */
    .recommendation-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
        box-shadow: 0 3px 10px rgba(0,0,0,0.05);
        transition: transform 0.2s ease;
        color: var(--dark);
    }

    .recommendation-card p,
    .recommendation-card li,
    .recommendation-card span,
    .recommendation-card div {
        color: inherit;
    }

    .recommendation-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
    }

    .recommendation-title {
        color: var(--primary);
        margin-top: 0;
        margin-bottom: 1rem;
        font-size: 1.2rem;
        font-weight: 600;
    }

    /* Секция Что-если - исправленная версия */
    .whatif-container {
        background-color: white;
        border: 1px solid var(--secondary-light);
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        color: var(--dark);
    }

    .whatif-controls {
        display: flex;
        align-items: flex-end;
        gap: 1rem;
        margin-top: 1rem;
    }

    .whatif-control-item {
        flex: 1;
    }

    /* Кнопки */
    .stButton button {
        background: linear-gradient(to right, var(--primary), var(--primary-light));
        color: white;
        border: none;
        padding: 0.7rem 1.3rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(42, 92, 138, 0.2);
    }

    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(42, 92, 138, 0.3);
    }

    /* Стили для ввода данных */
    .input-section {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #dee2e6;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        color: var(--dark);
    }

    .cohort-info-banner {
        background: linear-gradient(135deg, rgba(74, 139, 197, 0.1) 0%, rgba(91, 196, 159, 0.1) 100%);
        border: 1px solid var(--primary-light);
        border-radius: 10px;
        padding: 1rem 1.5rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 15px;
    }

    /* График */
    .plotly-graph {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        border: 1px solid #e9ecef;
    }

    /* Стили для таблицы истории */
    .dataframe-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin: 1rem 0;
    }

    /* Статистика в боковой панели */
    .stats-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin-bottom: 1rem;
        text-align: center;
        color: var(--dark);
    }

    .stats-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--primary);
    }

    .stats-label {
        color: #4f5b67;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }

    /* Пациент информация в результатах */
    .patient-info-container {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid #e9ecef;
        box-shadow: 0 3px 8px rgba(0,0,0,0.04);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 1rem;
        color: var(--dark);
    }

    .patient-info-item {
        display: flex;
        flex-direction: column;
        gap: 0.3rem;
    }

    .patient-info-label {
        color: #4f5b67;
        font-size: 0.85rem;
    }

    .patient-info-value {
        font-weight: 600;
        color: var(--primary);
        font-size: 1rem;
    }

    .history-detail-card {
        background: white;
        color: var(--dark);
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    .history-detail-title {
        color: var(--primary);
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .history-detail-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.8rem;
    }

    .history-detail-label {
        color: #5f6b7a;
        font-size: 0.85rem;
    }

    .history-detail-value {
        color: var(--dark);
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def parse_key_value_text(text: str) -> dict[str, object]:
    parsed_data: dict[str, object] = {}
    for line in text.splitlines():
        if '=' not in line:
            continue

        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        try:
            parsed_data[key] = float(value.replace(',', '.')) if '.' in value or ',' in value else int(value)
        except ValueError:
            parsed_data[key] = value

    return parsed_data


DRAFTS_DIR = Path("drafts")


def patient_data_to_text(patient_data: dict[str, object]) -> str:
    if not patient_data:
        return ""
    return "\n".join(f"{key} = {value}" for key, value in patient_data.items())


def set_input_buffers(patient_data: dict[str, object]) -> None:
    st.session_state.text_input_area = patient_data_to_text(patient_data)
    st.session_state.json_input_area = json.dumps(patient_data, ensure_ascii=False, indent=2)


def reset_input_widgets() -> None:
    st.session_state.form_widget_nonce = st.session_state.get("form_widget_nonce", 0) + 1


def get_draft_path(cohort: str, kind: str = "manual") -> Path:
    cohort_slug = get_model_cohort_name(cohort) or "patient"
    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "draft" if kind == "manual" else "autosave"
    return DRAFTS_DIR / f"{cohort_slug}_{suffix}.json"


def patient_data_signature(patient_data: dict[str, object]) -> str:
    return json.dumps(patient_data, ensure_ascii=False, sort_keys=True, default=str)


def patient_data_diff(current_data: dict[str, object], saved_data: dict[str, object] | None) -> dict[str, int]:
    saved_data = saved_data or {}
    current_keys = set(current_data)
    saved_keys = set(saved_data)
    added = len(current_keys - saved_keys)
    removed = len(saved_keys - current_keys)
    changed = sum(1 for key in current_keys & saved_keys if current_data.get(key) != saved_data.get(key))
    return {
        "added": added,
        "removed": removed,
        "changed": changed,
        "total": added + removed + changed,
    }


def save_draft(cohort: str, patient_data: dict[str, object], *, kind: str = "manual") -> Path:
    draft_path = get_draft_path(cohort, kind)
    payload = {
        "cohort": cohort,
        "kind": kind,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "signature": patient_data_signature(patient_data),
        "patient_data": patient_data,
    }
    draft_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return draft_path


def load_draft_payload(cohort: str, kind: str = "manual") -> dict[str, object] | None:
    draft_path = get_draft_path(cohort, kind)
    if not draft_path.exists():
        return None
    return json.loads(draft_path.read_text(encoding="utf-8"))


def load_draft(cohort: str, kind: str = "manual") -> dict[str, object] | None:
    payload = load_draft_payload(cohort, kind)
    if not payload:
        return None
    patient_data = payload.get("patient_data")
    return patient_data if isinstance(patient_data, dict) else None


def history_record_label(record: dict[str, object]) -> str:
    analysis_id = str(record.get("analysis_id") or record.get("id") or "A-UNKNOWN")
    patient_id = str(record.get("patient_id") or "P-UNKNOWN")
    return f"{analysis_id} | {patient_id}"


def matches_history_id(record_value: object, query: str) -> bool:
    value = str(record_value or "").strip().lower()
    query = query.strip().lower()
    if not query:
        return True
    return value.startswith(query) or value == query


def autosave_draft_if_needed(cohort: str, patient_data: dict[str, object]) -> tuple[bool, Path | None]:
    if not patient_data:
        return False, None

    current_signature = patient_data_signature(patient_data)
    autosave_payload = load_draft_payload(cohort, "autosave")
    if autosave_payload and autosave_payload.get("signature") == current_signature:
        return False, get_draft_path(cohort, "autosave")

    saved_path = save_draft(cohort, patient_data, kind="autosave")
    return True, saved_path

# ===================== ИНИЦИАЛИЗАЦИЯ СЕССИИ =====================
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'selected_cohort' not in st.session_state:
    st.session_state.selected_cohort = "Сепсис"
if 'show_history' not in st.session_state:
    st.session_state.show_history = False
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = {}
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'patient_id' not in st.session_state:
    st.session_state.patient_id = ""
if 'show_login' not in st.session_state:
    st.session_state.show_login = True
if 'show_register' not in st.session_state:
    st.session_state.show_register = False
if 'text_input_area' not in st.session_state:
    st.session_state.text_input_area = ""
if 'json_input_area' not in st.session_state:
    st.session_state.json_input_area = "{}"
if 'input_example_variant' not in st.session_state:
    st.session_state.input_example_variant = "Минимальный"
if 'input_buffer_profile_key' not in st.session_state:
    st.session_state.input_buffer_profile_key = ""
if 'show_examples' not in st.session_state:
    st.session_state.show_examples = False
if 'form_widget_nonce' not in st.session_state:
    st.session_state.form_widget_nonce = 0

# ===================== ПРОВЕРКА АУТЕНТИФИКАЦИИ =====================
# Проверка должна быть в самом начале, до всех остальных операций
if not st.session_state.authenticated:
    # Login/Register page
    st.markdown("""
    <div class="main-header">
        <h1 style="margin:0; font-size:2rem; font-weight:600;">Immunorisk Studio</h1>
        <p style="margin:0; opacity:0.9; font-size:0.95rem;">
            Система интеллектуального моделирования иммунного ответа
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Вход", "Регистрация"])
    
    with tab1:
        st.markdown("### Вход в систему")
        login_username = st.text_input("Имя пользователя", key="login_username")
        login_password = st.text_input("Пароль", type="password", key="login_password")
        
        if st.button("Войти", width="stretch", type="primary"):
            success, user_data, message = login_user(login_username, login_password)
            if success:
                st.session_state.authenticated = True
                st.session_state.current_user = user_data
                st.success(message)
                st.rerun()
            else:
                st.error(message)
    
    with tab2:
        st.markdown("### Регистрация нового пользователя")
        reg_username = st.text_input("Имя пользователя", key="reg_username")
        reg_password = st.text_input("Пароль", type="password", key="reg_password")
        reg_full_name = st.text_input("ФИО", key="reg_full_name")
        reg_specialization = st.text_input("Специализация", key="reg_specialization", 
                                          value="Иммунолог-инфекционист")
        
        if st.button("Зарегистрироваться", width="stretch", type="primary"):
            success, message = register_user(reg_username, reg_password, reg_full_name, reg_specialization)
            if success:
                st.success(message)
                st.info("Теперь вы можете войти в систему")
            else:
                st.error(message)

    # Восстановление пароля через администратора / секретный код
    with st.expander("Восстановление пароля (администратор)"):
        st.markdown("Если вы забыли пароль, администратор может сбросить его по секретному коду.")
        reset_username = st.text_input("Имя пользователя для сброса", key="reset_username")
        admin_code = st.text_input("Секретный код администратора", type="password", key="reset_admin_code")
        reset_new_pwd = st.text_input("Новый пароль", type="password", key="reset_new_pwd")
        reset_new_pwd_confirm = st.text_input("Повторите новый пароль", type="password", key="reset_new_pwd_confirm")

        if st.button("Сбросить пароль", width="stretch", key="reset_pwd_button"):
            if not reset_username or not admin_code or not reset_new_pwd or not reset_new_pwd_confirm:
                st.error("Пожалуйста, заполните все поля для сброса пароля.")
            elif reset_new_pwd != reset_new_pwd_confirm:
                st.error("Новый пароль и подтверждение не совпадают.")
            else:
                success, message = admin_reset_password(admin_code, reset_username, reset_new_pwd)
                if success:
                    st.success(message)
                    st.info("Теперь пользователь может войти с новым паролем.")
                else:
                    st.error(message)

    st.stop()  # Останавливаем выполнение, если не аутентифицирован

# ===================== ДАННЫЕ КОГОРТ =====================
# Get available cohorts with ML models
available_cohorts = get_available_cohorts()

cohorts = {
    "Сепсис": {
        "target": "Шкала SOFA",
        "patients": 248,
        "features": 42,
        "type": "Регрессия",
        "description": "Оценка тяжести сепсиса по доступным клинико-лабораторным данным",
        "available": True
    },
    "Перитонит": {
        "target": "Шкала SOFA",
        "patients": 187,
        "features": 38,
        "type": "Регрессия",
        "description": "Оценка тяжести перитонита и прогноз осложнений",
        "available": True
    },
    "ИБС": {
        "target": "Резистентность",
        "patients": 312,
        "features": 45,
        "type": "Бинарная",
        "description": "Прогноз резистентности к стандартной терапии",
        "available": True
    },
    "IL-2 постковид": {
        "target": "Постковид",
        "patients": 165,
        "features": 58,
        "type": "Многоклассовая",
        "description": "Классификация постковидных состояний после IL-2 терапии",
        "available": True
    },
    "COVID-19": {
        "target": "Тяжесть течения",
        "patients": 423,
        "features": 52,
        "type": "Регрессия",
        "description": "Прогноз тяжести COVID-19 и риска осложнений",
        "available": False
    },
    "Ревматоидный артрит": {
        "target": "Активность",
        "patients": 198,
        "features": 41,
        "type": "Регрессия",
        "description": "Оценка активности заболевания и ответа на терапию",
        "available": False
    },
    "Онкология": {
        "target": "Ответ на иммунотерапию",
        "patients": 267,
        "features": 63,
        "type": "Бинарная",
        "description": "Прогноз ответа на иммунотерапию при онкозаболеваниях",
        "available": False
    },
}

# Update cohort info with model data if available
for cohort_name in available_cohorts:
    model_info = get_model_info(cohort_name)
    if model_info:
        if cohort_name in cohorts:
            cohorts[cohort_name]["features"] = model_info["n_features"]
            task_type_map = {
                "regression": "Регрессия",
                "classification": "Бинарная",
                "multiclass": "Многоклассовая"
            }
            cohorts[cohort_name]["type"] = task_type_map.get(model_info["task"], cohorts[cohort_name]["type"])

# ===================== ЛЕВАЯ ПАНЕЛЬ =====================
with st.sidebar:
    # Карточка пользователя (пользователь уже проверен выше при аутентификации)
    user = st.session_state.current_user
    # Пользователь должен быть здесь, так как мы проверили аутентификацию выше
    
    initials = "".join([n[0].upper() for n in user.get("full_name", "Пользователь").split()[:2]]) if user and user.get("full_name") else "П"
    
    st.markdown(f"""
    <div class="user-card">
        <div style="width: 70px; height: 70px; background: linear-gradient(135deg, var(--primary), var(--primary-light)); 
                    border-radius: 50%; display: flex; align-items: center; justify-content: center; 
                    color: white; font-weight: bold; font-size: 1.5rem; margin: 0 auto 1rem auto;">
            {initials}
        </div>
        <div style="font-weight: bold; color: var(--primary); font-size: 1.1rem; margin-bottom: 0.3rem;">
            {user.get("full_name", "Пользователь") if user else "Пользователь"}
        </div>
        <div style="color: #4f5b67; font-size: 0.9rem; margin-bottom: 0.5rem;">
            {user.get("specialization", "Врач") if user else "Врач"}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Статистика пользователя
    user_stats = get_user_stats(user.get("username", "") if user else "")
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-value">{user_stats.get('total_patients', 0)}</div>
            <div class="stats-label">Пациентов</div>
        </div>
        """, unsafe_allow_html=True)
    with col_stat2:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-value">{user_stats.get('total_analyses', 0)}</div>
            <div class="stats-label">Анализов</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Смена пароля
    with st.expander("Сменить пароль"):
        st.markdown("Введите текущий и новый пароль для смены учетных данных.")
        current_pwd = st.text_input("Текущий пароль", type="password", key="change_pwd_current")
        new_pwd = st.text_input("Новый пароль", type="password", key="change_pwd_new")
        new_pwd_confirm = st.text_input("Повторите новый пароль", type="password", key="change_pwd_confirm")

        if st.button("Изменить пароль", width="stretch"):
            if not current_pwd or not new_pwd or not new_pwd_confirm:
                st.error("Пожалуйста, заполните все поля для смены пароля.")
            elif new_pwd != new_pwd_confirm:
                st.error("Новый пароль и подтверждение не совпадают.")
            else:
                success, message = change_password(user.get("username", ""), current_pwd, new_pwd)
                if success:
                    st.success(message)
                else:
                    st.error(message)

    st.markdown("---")

    if st.button("Выйти", width="stretch"):
        st.session_state.authenticated = False
        st.session_state.current_user = None
        st.session_state.prediction_made = False
        st.session_state.show_history = False
        st.session_state.show_examples = False
        st.rerun()

    st.divider()

    # Выбор когорты
    st.markdown("### Выбор когорты")

    # Filter to show only available cohorts (those with trained models)
    available_cohort_list = get_available_cohorts()
    if not available_cohort_list:
        st.error("Нет доступных моделей. Пожалуйста, обучите модели сначала.")
        st.stop()
    
    # Filter cohorts dict to only include available ones
    cohorts = {k: v for k, v in cohorts.items() if k in available_cohort_list}
    
    # Ensure selected cohort is in available list
    if st.session_state.selected_cohort not in available_cohort_list:
        st.session_state.selected_cohort = available_cohort_list[0]
        st.session_state.prediction_made = False
        st.session_state.prediction_result = None

    selected_cohort = st.selectbox(
        "Выберите когорту для анализа:",
        available_cohort_list,
        index=available_cohort_list.index(st.session_state.selected_cohort) if st.session_state.selected_cohort in available_cohort_list else 0,
        label_visibility="collapsed"
    )

    if selected_cohort != st.session_state.selected_cohort:
        st.session_state.selected_cohort = selected_cohort
        st.session_state.prediction_made = False
        st.session_state.prediction_result = None
        st.session_state.patient_data = {}
        set_input_buffers({})
        reset_input_widgets()
        st.session_state.show_examples = False
        st.rerun()

    # Информация о выбранной когорте (all shown cohorts have models)
    cohort_info = cohorts[st.session_state.selected_cohort]
    
    st.info(f"""
    **{st.session_state.selected_cohort}**  
    {cohort_info['description']}  

    **Тип:** {cohort_info['type']}  
    **Пациентов:** {cohort_info['patients']}  
    **Признаков:** {cohort_info['features']}
    
    ✅ Модель доступна
    """)

    st.divider()

    # Быстрые действия
    st.markdown("### Быстрые действия")

    if st.button("Новый анализ", width="stretch", type="primary"):
        st.session_state.prediction_made = False
        st.session_state.show_history = False
        st.session_state.show_examples = False

    if st.button("История анализов", width="stretch"):
        st.session_state.show_history = True
        st.session_state.prediction_made = True

    if st.button("Библиотека примеров", width="stretch"):
        st.session_state.show_examples = True
        st.session_state.show_history = False
        st.session_state.prediction_made = False
        st.rerun()

# Get user's personal history (only if authenticated)
user = st.session_state.current_user
if user and "username" in user:
    history_data = get_user_history(user["username"])
else:
    history_data = []

# ===================== ОСНОВНОЙ КОНТЕНТ =====================

# Заголовок
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size:2rem; font-weight:600;">Immunorisk Studio</h1>
    <p style="margin:0; opacity:0.9; font-size:0.95rem;">
        Система интеллектуального моделирования иммунного ответа
    </p>
</div>
""", unsafe_allow_html=True)

# ===================== ИСТОРИЯ АНАЛИЗОВ =====================
if st.session_state.show_history:
    st.markdown('<div class="section-title">История анализов</div>', unsafe_allow_html=True)

    # Фильтры для истории
    col_filter1, col_filter2, col_filter3, col_filter4, col_filter5 = st.columns(5)

    with col_filter1:
        patient_search_query = st.text_input("Поиск по ID пациента", placeholder="Например: P-002")

    with col_filter2:
        analysis_search_query = st.text_input("Поиск по ID анализа", placeholder="Например: A-002")

    with col_filter3:
        cohort_filter = st.multiselect(
            "Фильтр по когорте",
            ["Все", "Сепсис", "Перитонит", "ИБС", "COVID-19", "Онкология", "IL-2 постковид", "Ревматоидный артрит"],
            default=["Все"]
        )

    with col_filter4:
        risk_filter = st.multiselect(
            "Фильтр по риску",
            ["Все", "Высокий", "Средний", "Низкий"],
            default=["Все"]
        )

    with col_filter5:
        date_range = st.date_input(
            "Период",
            [datetime.now() - timedelta(days=30), datetime.now()]
        )

    # Применяем фильтры
    filtered_history = history_data.copy()

    if patient_search_query:
        filtered_history = [
            h for h in filtered_history
            if matches_history_id(h.get("patient_id", ""), patient_search_query)
        ]

    if analysis_search_query:
        filtered_history = [
            h for h in filtered_history
            if matches_history_id(h.get("analysis_id", h.get("id", "")), analysis_search_query)
        ]

    if "Все" not in cohort_filter:
        filtered_history = [h for h in filtered_history if h["cohort"] in cohort_filter]

    if "Все" not in risk_filter:
        filtered_history = [h for h in filtered_history if h["risk"] in risk_filter]

    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_history = [
            h for h in filtered_history
            if start_date <= datetime.strptime(h["date"], "%d.%m.%Y %H:%M").date() <= end_date
        ]

    # Сортировка
    sort_option = st.selectbox("Сортировка", ["Дата (новые сначала)", "Дата (старые сначала)", "Риск", "SOFA"])

    if sort_option == "Дата (новые сначала)":
        filtered_history.sort(key=lambda x: datetime.strptime(x["date"], "%d.%m.%Y %H:%M"), reverse=True)
    elif sort_option == "Дата (старые сначала)":
        filtered_history.sort(key=lambda x: datetime.strptime(x["date"], "%d.%m.%Y %H:%M"))
    elif sort_option == "Риск":
        risk_order = {"Высокий": 0, "Средний": 1, "Низкий": 2}
        filtered_history.sort(key=lambda x: risk_order.get(x["risk"], 3))
    elif sort_option == "SOFA":
        filtered_history.sort(key=lambda x: x["sofa"], reverse=True)

    # Отображение таблицы через st.dataframe
    if filtered_history:
        st.markdown(f"**Найдено записей:** {len(filtered_history)}")

        # Создаем DataFrame
        history_df = pd.DataFrame(filtered_history)
        for column in ["id", "prediction"]:
            if column in history_df.columns:
                history_df = history_df.drop(columns=[column])

        display_columns = [
            column for column in ["analysis_id", "patient_id", "cohort", "risk", "sofa", "doctor", "status", "date"]
            if column in history_df.columns
        ]
        history_df = history_df[display_columns]

        # Создаем стилизованный DataFrame с закругленными углами
        st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)

        # Используем st.dataframe с кастомным CSS
        st.dataframe(
            history_df,
            column_config={
                "analysis_id": st.column_config.TextColumn("ID анализа", width="small"),
                "patient_id": st.column_config.TextColumn("ID пациента", width="small"),
                "date": st.column_config.TextColumn("Дата анализа", width="medium"),
                "cohort": st.column_config.TextColumn("Когорта", width="small"),
                "risk": st.column_config.TextColumn(
                    "Риск",
                    help="Уровень риска",
                    width="small"
                ),
                "sofa": st.column_config.NumberColumn(
                    "SOFA",
                    help="Шкала SOFA",
                    format="%.1f",
                    width="small"
                ),
                "doctor": st.column_config.TextColumn("Врач", width="medium"),
                "status": st.column_config.TextColumn("Статус", width="medium")
            },
            hide_index=True,
            width="stretch"
        )

        st.markdown('</div>', unsafe_allow_html=True)

        # Действия с выбранными записями
        st.divider()
        st.markdown("### Действия с записями")

        col_action1, col_action2, col_action3 = st.columns(3)

        with col_action1:
            history_options = [""] + [history_record_label(h) for h in filtered_history]
            selected_id = st.selectbox("Выберите запись для детального просмотра", history_options)
            if selected_id:
                selected_analysis_id = selected_id.split(" | ", 1)[0]
                selected_record = next(
                    (h for h in filtered_history if str(h.get("analysis_id", h.get("id", ""))) == selected_analysis_id),
                    None,
                )
                if selected_record:
                    # Определяем цвет для риска
                    risk_color = ""
                    if selected_record["risk"] == "Высокий":
                        risk_color = "#e63946"
                    elif selected_record["risk"] == "Средний":
                        risk_color = "#d39e00"
                    else:
                        risk_color = "#28a745"

                    st.markdown(f"""
                    <div class="history-detail-card">
                        <div class="history-detail-title">Детали записи:</div>
                        <div class="history-detail-grid">
                            <div>
                                <div class="history-detail-label">Пациент</div>
                                <div class="history-detail-value">{selected_record.get('patient_id', 'P-UNKNOWN')}</div>
                            </div>
                            <div>
                                <div class="history-detail-label">Анализ</div>
                                <div class="history-detail-value">{selected_record.get('analysis_id', selected_record.get('id', 'A-UNKNOWN'))}</div>
                            </div>
                            <div>
                                <div class="history-detail-label">Дата</div>
                                <div class="history-detail-value">{selected_record['date']}</div>
                            </div>
                            <div>
                                <div class="history-detail-label">Когорта</div>
                                <div class="history-detail-value">{selected_record['cohort']}</div>
                            </div>
                            <div>
                                <div class="history-detail-label">Риск</div>
                                <div class="history-detail-value" style="color: {risk_color};">{selected_record['risk']}</div>
                            </div>
                            <div>
                                <div class="history-detail-label">SOFA</div>
                                <div class="history-detail-value">{selected_record['sofa']}</div>
                            </div>
                            <div>
                                <div class="history-detail-label">Врач</div>
                                <div class="history-detail-value">{selected_record['doctor']}</div>
                            </div>
                            <div style="grid-column: span 2;">
                                <div class="history-detail-label">Статус</div>
                                <div class="history-detail-value">{selected_record['status']}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        with col_action2:
            st.markdown("###")
            # Generate CSV content
            if filtered_history:
                csv_content = generate_csv_history(filtered_history)
                csv_filename = f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                
                st.download_button(
                    label="Экспорт в CSV",
                    data=csv_content,
                    file_name=csv_filename,
                    mime="text/csv",
                    width="stretch"
                )
            else:
                st.info("Нет данных для экспорта")

        with col_action3:
            st.markdown("###")
            if st.button("Очистить фильтры", width="stretch"):
                st.session_state.show_history = True
                st.rerun()

        clear_col1, clear_col2 = st.columns([1, 2])
        with clear_col1:
            if st.button("Очистить историю", width="stretch", type="secondary"):
                success, message = clear_user_history(user["username"])
                if success:
                    st.session_state.current_user = {
                        **user,
                        "stats": get_user_stats(user["username"]),
                        "history": get_user_history(user["username"]),
                    }
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
        with clear_col2:
            st.caption("Очищает историю анализов текущего пользователя и сбрасывает связанные счетчики.")

    else:
        st.warning("Записи не найдены по заданным фильтрам")

        if st.button("Очистить историю", width="stretch"):
            success, message = clear_user_history(user["username"])
            if success:
                st.session_state.current_user = {
                    **user,
                    "stats": get_user_stats(user["username"]),
                    "history": get_user_history(user["username"]),
                }
                st.success(message)
                st.rerun()
            else:
                st.error(message)

    # Кнопка возврата
    if st.button("Вернуться к анализу", width="stretch"):
        st.session_state.show_history = False
        st.rerun()

# ===================== БИБЛИОТЕКА ПРИМЕРОВ =====================
elif st.session_state.show_examples:
    st.markdown('<div class="section-title">Библиотека примеров пациентов</div>', unsafe_allow_html=True)
    st.caption("Здесь можно быстро загрузить минимальный или расширенный учебный кейс для любой доступной когорты.")

    selected_library_cohort = st.selectbox(
        "Когорта для примеров",
        list(cohorts.keys()),
        index=list(cohorts.keys()).index(st.session_state.selected_cohort) if st.session_state.selected_cohort in cohorts else 0,
    )

    lib_col1, lib_col2 = st.columns(2)
    for col, variant_label, variant_key in [
        (lib_col1, "Минимальный пример", "minimal"),
        (lib_col2, "Расширенный пример", "extended"),
    ]:
        with col:
            example = get_input_examples(selected_library_cohort, variant_key)
            example_data = json.loads(example["json"])
            st.markdown(f"### {variant_label}")
            st.code(example["text"], language="text")
            if st.button(f"Загрузить: {variant_label}", key=f"library_{selected_library_cohort}_{variant_key}", width="stretch"):
                st.session_state.selected_cohort = selected_library_cohort
                st.session_state.input_example_variant = "Расширенный" if variant_key == "extended" else "Минимальный"
                st.session_state.patient_data = example_data
                st.session_state.patient_id = str(example_data.get("patient_id", "P-001"))
                set_input_buffers(example_data)
                reset_input_widgets()
                st.session_state.input_buffer_profile_key = f"{selected_library_cohort}:{variant_key}"
                st.session_state.show_examples = False
                st.session_state.show_history = False
                st.session_state.prediction_made = False
                st.rerun()

    if st.button("Вернуться к вводу пациента", width="stretch"):
        st.session_state.show_examples = False
        st.rerun()

# ===================== РАЗДЕЛ: АНАЛИЗ ПАЦИЕНТА =====================
elif not st.session_state.prediction_made:
    st.markdown('<div class="section-title">Ввод данных пациента</div>', unsafe_allow_html=True)

    # Информация о выбранной когорте
    st.markdown(f"""
    <div class="cohort-info-banner">
        <div style="font-weight: 600; color: var(--primary);">Выбрана когорта:</div>
        <div style="font-size: 1.1rem; font-weight: 600;">{st.session_state.selected_cohort}</div>
        <div style="color: #4f5b67; font-size: 0.9rem;">{cohorts[st.session_state.selected_cohort]['description']}</div>
    </div>
    """, unsafe_allow_html=True)

    # Выбор метода ввода
    input_method = st.radio(
        "Способ ввода данных",
        ["Форма", "Текст", "JSON", "Файл"],
        horizontal=True,
        label_visibility="collapsed"
    )

    example_variant_label = st.radio(
        "Тип примера",
        ["Минимальный", "Расширенный"],
        horizontal=True,
        key="input_example_variant",
    )
    example_variant = "extended" if example_variant_label == "Расширенный" else "minimal"

    input_examples = get_input_examples(st.session_state.selected_cohort, example_variant)
    example_payload = json.loads(input_examples["json"])
    current_buffer_profile_key = f"{st.session_state.selected_cohort}:{example_variant}"
    if st.session_state.input_buffer_profile_key != current_buffer_profile_key and not st.session_state.patient_data:
        st.session_state.input_buffer_profile_key = current_buffer_profile_key

    current_input_json = json.dumps(st.session_state.patient_data, ensure_ascii=False, indent=2) if st.session_state.patient_data else "{}"
    current_input_csv = pd.DataFrame([st.session_state.patient_data]).to_csv(index=False) if st.session_state.patient_data else ""
    cohort_slug = get_model_cohort_name(st.session_state.selected_cohort) or "patient"
    manual_draft_path = get_draft_path(st.session_state.selected_cohort, "manual")
    autosave_path = get_draft_path(st.session_state.selected_cohort, "autosave")
    draft_exists = manual_draft_path.exists()
    autosave_exists = autosave_path.exists()
    manual_draft_data = load_draft(st.session_state.selected_cohort, "manual")
    autosave_payload = load_draft_payload(st.session_state.selected_cohort, "autosave")
    manual_diff = patient_data_diff(st.session_state.patient_data, manual_draft_data)

    st.markdown("### Быстрые действия с вводом")
    action_col1, action_col2, action_col3, action_col4, action_col5, action_col6 = st.columns(6)

    with action_col1:
        if st.button("Загрузить пример", width="stretch"):
            st.session_state.patient_data = example_payload.copy()
            st.session_state.patient_id = example_payload.get("patient_id", "P-001")
            set_input_buffers(example_payload)
            reset_input_widgets()
            st.session_state.input_buffer_profile_key = current_buffer_profile_key
            st.success("Пример для выбранной когорты загружен в форму")
            st.rerun()

    with action_col2:
        if st.button("Очистить ввод", width="stretch"):
            st.session_state.patient_data = {}
            st.session_state.patient_id = ""
            set_input_buffers({})
            reset_input_widgets()
            st.session_state.input_buffer_profile_key = current_buffer_profile_key
            st.success("Текущий ввод очищен")
            st.rerun()

    with action_col3:
        if st.button("Сохранить черновик", width="stretch", disabled=not bool(st.session_state.patient_data)):
            saved_path = save_draft(st.session_state.selected_cohort, st.session_state.patient_data, kind="manual")
            st.success(f"Черновик сохранён: {saved_path}")

    with action_col4:
        if st.button("Загрузить черновик", width="stretch", disabled=not draft_exists):
            draft_data = load_draft(st.session_state.selected_cohort, "manual")
            if draft_data is not None:
                st.session_state.patient_data = draft_data
                st.session_state.patient_id = str(draft_data.get("patient_id", "P-001"))
                set_input_buffers(draft_data)
                reset_input_widgets()
                st.session_state.input_buffer_profile_key = current_buffer_profile_key
                st.success("Черновик загружен")
                st.rerun()
            else:
                st.error("Не удалось прочитать черновик")

    with action_col5:
        st.download_button(
            label="Экспорт JSON",
            data=current_input_json,
            file_name=f"{cohort_slug}_input.json",
            mime="application/json",
            width="stretch",
            disabled=not bool(st.session_state.patient_data),
        )

    with action_col6:
        st.download_button(
            label="Экспорт CSV",
            data=current_input_csv,
            file_name=f"{cohort_slug}_input.csv",
            mime="text/csv",
            width="stretch",
            disabled=not bool(st.session_state.patient_data),
        )

    if draft_exists:
        st.caption(f"Ручной черновик: {manual_draft_path}")
    elif st.session_state.patient_data:
        st.caption("Ручной черновик ещё не сохранён")

    if manual_diff["total"] == 0 and draft_exists:
        st.success("Текущие данные совпадают с последним ручным сохранением")
    elif st.session_state.patient_data and draft_exists:
        st.info(
            f"Изменения с прошлого ручного сохранения: +{manual_diff['added']} / ~{manual_diff['changed']} / -{manual_diff['removed']}"
        )

    if autosave_exists and autosave_payload:
        st.caption(f"Автосохранение: {autosave_payload.get('saved_at', 'неизвестно')} -> {autosave_path}")

    if input_method == "Форма":
        # Dynamic form based on cohort model features
        st.session_state.patient_data = generate_dynamic_form(
            st.session_state.selected_cohort,
            st.session_state.patient_data,
            st.session_state.form_widget_nonce,
        )
        set_input_buffers(st.session_state.patient_data)
        if "patient_id" in st.session_state.patient_data:
            st.session_state.patient_id = st.session_state.patient_data["patient_id"]

    elif input_method == "Текст":
        text_input = st.text_area(
            "Введите данные пациента",
            height=250,
            key="text_input_area"
        )
        if text_input:
            parsed_data = parse_key_value_text(text_input)
            if parsed_data:
                st.session_state.patient_data = parsed_data.copy()
                st.session_state.input_buffer_profile_key = current_buffer_profile_key
                if "patient_id" in parsed_data:
                    st.session_state.patient_id = parsed_data["patient_id"]

    elif input_method == "JSON":
        json_input = st.text_area(
            "Введите данные пациента в формате JSON",
            height=250,
            key="json_input_area"
        )
        # Parse JSON input
        if json_input:
            try:
                parsed_data = json.loads(json_input)
                if parsed_data:
                    st.session_state.patient_data = parsed_data.copy()
                    st.session_state.input_buffer_profile_key = current_buffer_profile_key
                    if "patient_id" in parsed_data:
                        st.session_state.patient_id = parsed_data["patient_id"]
            except json.JSONDecodeError as e:
                st.error(f"Ошибка парсинга JSON: {str(e)}")

    else:  # Файл
        uploaded_file = st.file_uploader(
            "Загрузите файл с данными пациента",
            type=['csv', 'txt', 'json', 'xlsx']
        )
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df_file = pd.read_csv(uploaded_file)
                    if not df_file.empty:
                        # Convert first row to dict
                        parsed_data = df_file.iloc[0].to_dict()
                        st.session_state.patient_data.update(parsed_data)
                        st.session_state.input_buffer_profile_key = current_buffer_profile_key
                        set_input_buffers(st.session_state.patient_data)
                        st.success(f"Файл '{uploaded_file.name}' успешно загружен")
                elif uploaded_file.name.endswith('.xlsx'):
                    df_file = pd.read_excel(uploaded_file)
                    if not df_file.empty:
                        parsed_data = df_file.iloc[0].to_dict()
                        st.session_state.patient_data.update(parsed_data)
                        st.session_state.input_buffer_profile_key = current_buffer_profile_key
                        set_input_buffers(st.session_state.patient_data)
                        if "patient_id" in parsed_data:
                            st.session_state.patient_id = parsed_data["patient_id"]
                        st.success(f"Файл '{uploaded_file.name}' успешно загружен")
                elif uploaded_file.name.endswith('.txt'):
                    parsed_data = parse_key_value_text(uploaded_file.getvalue().decode('utf-8'))
                    if parsed_data:
                        st.session_state.patient_data.update(parsed_data)
                        st.session_state.input_buffer_profile_key = current_buffer_profile_key
                        set_input_buffers(st.session_state.patient_data)
                        if "patient_id" in parsed_data:
                            st.session_state.patient_id = parsed_data["patient_id"]
                        st.success(f"Файл '{uploaded_file.name}' успешно загружен")
                elif uploaded_file.name.endswith('.json'):
                    parsed_data = json.load(uploaded_file)
                    if parsed_data:
                        st.session_state.patient_data.update(parsed_data)
                        st.session_state.input_buffer_profile_key = current_buffer_profile_key
                        set_input_buffers(st.session_state.patient_data)
                        if "patient_id" in parsed_data:
                            st.session_state.patient_id = parsed_data["patient_id"]
                        st.success(f"Файл '{uploaded_file.name}' успешно загружен")
            except Exception as e:
                st.error(f"Ошибка при загрузке файла: {str(e)}")

    autosave_saved, autosave_saved_path = autosave_draft_if_needed(
        st.session_state.selected_cohort,
        st.session_state.patient_data,
    )
    latest_manual_draft = load_draft(st.session_state.selected_cohort, "manual")
    latest_manual_diff = patient_data_diff(st.session_state.patient_data, latest_manual_draft)

    if autosave_saved and autosave_saved_path:
        st.caption(f"Автосохранение обновлено: {autosave_saved_path}")

    if latest_manual_diff["total"] == 0 and latest_manual_draft:
        st.success("Изменений относительно последнего ручного сохранения нет")
    elif st.session_state.patient_data and latest_manual_draft:
        st.info(
            f"Новые изменения относительно ручного черновика: +{latest_manual_diff['added']} / ~{latest_manual_diff['changed']} / -{latest_manual_diff['removed']}"
        )
    elif st.session_state.patient_data:
        st.info("Можно сохранить текущий кейс как ручной черновик для дальнейшей работы")

    profile_status = get_minimal_profile_status(st.session_state.selected_cohort, st.session_state.patient_data)
    if st.session_state.patient_data:
        missing_profile_preview = ", ".join(profile_status["missing_fields"][:5])
        if profile_status["completion"] < 0.6:
            st.warning(
                f"Минимальный профиль заполнен на {profile_status['completion'] * 100:.1f}% "
                f"({len(profile_status['filled_fields'])} из {profile_status['total_fields']} полей)."
            )
        else:
            st.success(
                f"Минимальный профиль заполнен на {profile_status['completion'] * 100:.1f}% "
                f"({len(profile_status['filled_fields'])} из {profile_status['total_fields']} полей)."
            )

        if missing_profile_preview:
            st.caption(f"Поля минимального профиля, которые ещё стоит проверить: {missing_profile_preview}")

    coverage_info = get_input_coverage(st.session_state.patient_data, st.session_state.selected_cohort)
    total_features = coverage_info["total_features"]
    filled_count = coverage_info["filled_count"]
    coverage_ratio = coverage_info["coverage"]

    if total_features and st.session_state.patient_data:
        missing_preview = ", ".join(coverage_info["missing_features"][:5])
        if coverage_ratio < 0.15:
            st.warning(
                f"Заполнено только {filled_count} из {total_features} признаков "
                f"({coverage_ratio * 100:.1f}%). Прогноз будет очень грубым приближением."
            )
        elif coverage_ratio < 0.4:
            st.info(
                f"Заполнено {filled_count} из {total_features} признаков "
                f"({coverage_ratio * 100:.1f}%). Для более надежного результата заполните больше полей."
            )
        elif st.session_state.patient_data:
            st.success(
                f"Заполнено {filled_count} из {total_features} признаков "
                f"({coverage_ratio * 100:.1f}%)."
            )

        if missing_preview:
            st.caption(f"Примеры незаполненных признаков: {missing_preview}")

    # Кнопка запуска анализа
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        if st.button("Запустить анализ пациента", width="stretch", type="primary"):
            # Model should always be available since we filtered cohorts
            if st.session_state.selected_cohort not in available_cohort_list:
                st.error(f"Модель не доступна для когорты '{st.session_state.selected_cohort}'. Выберите другую когорту.")
            else:
                # Make prediction using ML service
                try:
                    with st.spinner("Выполняется анализ..."):
                        prediction_result = predict_patient(
                            st.session_state.patient_data.copy(),  # Use copy to ensure fresh data
                            st.session_state.selected_cohort
                        )
                        st.session_state.prediction_result = prediction_result
                        st.session_state.prediction_made = True
                        
                        # Calculate risk level based on prediction
                        risk_level = "Низкий"
                        if prediction_result["task"] == "regression":
                            pred_value = prediction_result["pred"][0] if prediction_result["pred"] else 0
                            if pred_value >= 8:
                                risk_level = "Высокий"
                            elif pred_value >= 5:
                                risk_level = "Средний"
                        elif prediction_result["task"] == "classification":
                            proba = prediction_result.get("proba", [0])[0] if prediction_result.get("proba") else 0
                            if proba >= 0.7:
                                risk_level = "Высокий"
                            elif proba >= 0.4:
                                risk_level = "Средний"
                        
                        # Save to user's personal history
                        user = st.session_state.current_user
                        sofa_value = st.session_state.patient_data.get("sofa", 7)
                        if prediction_result["task"] == "regression" and prediction_result["pred"]:
                            sofa_value = prediction_result["pred"][0]
                        
                        analysis_record = {
                            "patient_id": str(st.session_state.patient_data.get("patient_id") or st.session_state.patient_id or "P-UNKNOWN"),
                            "cohort": st.session_state.selected_cohort,
                            "risk": risk_level,
                            "sofa": sofa_value,
                            "doctor": user["full_name"],
                            "status": "Завершен",
                            "prediction": prediction_result
                        }
                        
                        add_to_user_history(user["username"], analysis_record)
                        update_user_stats(user["username"], increment_patients=1, increment_analyses=1)
                        
                        # Refresh user data in session
                        st.session_state.current_user = {
                            **user,
                            "stats": get_user_stats(user["username"]),
                            "history": get_user_history(user["username"])
                        }
                        
                        st.success("Анализ выполнен успешно!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Ошибка при выполнении анализа: {str(e)}")
                    st.code(traceback.format_exc())
                    st.session_state.prediction_made = False

# ===================== РАЗДЕЛ: РЕЗУЛЬТАТЫ АНАЛИЗА =====================
else:
    st.markdown('<div class="section-title">Анализ и результаты</div>', unsafe_allow_html=True)

    # Check if we have prediction results
    if st.session_state.prediction_result is None:
        st.warning("Результаты анализа не найдены. Пожалуйста, выполните анализ заново.")
        st.session_state.prediction_made = False
        st.rerun()
    
    pred_result = st.session_state.prediction_result
    patient_data = st.session_state.patient_data
    
    # Calculate risk level
    risk_level = "Низкий"
    risk_class = "risk-low"
    if pred_result["task"] == "regression":
        pred_value = pred_result["pred"][0] if pred_result["pred"] else 0
        if pred_value >= 8:
            risk_level = "Высокий"
            risk_class = "risk-high"
        elif pred_value >= 5:
            risk_level = "Средний"
            risk_class = "risk-medium"
    elif pred_result["task"] == "classification":
        proba = pred_result.get("proba", [0])[0] if pred_result.get("proba") else 0
        if proba >= 0.7:
            risk_level = "Высокий"
            risk_class = "risk-high"
        elif proba >= 0.4:
            risk_level = "Средний"
            risk_class = "risk-medium"

    # Информация о пациенте в красивом контейнере
    st.markdown(f"""
    <div class="patient-info-container">
        <div class="patient-info-item">
            <div class="patient-info-label">Пациент</div>
            <div class="patient-info-value">{st.session_state.patient_id}</div>
        </div>
        <div class="patient-info-item">
            <div class="patient-info-label">Дата анализа</div>
            <div class="patient-info-value">{datetime.now().strftime('%d.%m.%Y %H:%M')}</div>
        </div>
        <div class="patient-info-item">
            <div class="patient-info-label">Когорта</div>
            <div class="patient-info-value">{st.session_state.selected_cohort}</div>
        </div>
        <div style="margin-left: auto;">
            <div class="{risk_class} risk-badge">{risk_level} риск</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ===================== БЛОК: ОСНОВНЫЕ ПОКАЗАТЕЛИ =====================
    st.markdown('<div class="subsection-title">Основные показатели прогноза</div>', unsafe_allow_html=True)

    col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)

    # Get model info for metrics
    model_info = get_model_info(st.session_state.selected_cohort)
    metrics = model_info["metrics"] if model_info else {}
    
    # Metric 1: Prediction value
    if pred_result["task"] == "regression":
        pred_value = pred_result["pred"][0] if pred_result["pred"] else 0
        current_sofa = patient_data.get("sofa", 0)
        delta = pred_value - current_sofa
        delta_class = "delta-negative" if delta > 0 else "delta-positive"
        delta_sign = "+" if delta > 0 else ""
        
        with col_metrics1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Прогноз {cohorts[st.session_state.selected_cohort]['target']}</div>
                <div class="metric-value">{pred_value:.1f}</div>
                <div class="metric-delta {delta_class}">{delta_sign}{delta:.1f} от исходного</div>
            </div>
            """, unsafe_allow_html=True)
    elif pred_result["task"] == "classification":
        proba = pred_result.get("proba", [0])[0] if pred_result.get("proba") else 0
        pred_binary = pred_result.get("pred", [0])[0] if pred_result.get("pred") else 0
        
        with col_metrics1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Вероятность</div>
                <div class="metric-value">{proba*100:.1f}%</div>
                <div class="metric-delta {'delta-negative' if proba > 0.5 else 'delta-positive'}">
                    {'Положительный' if pred_binary == 1 else 'Отрицательный'}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:  # multiclass
        top_pred = pred_result.get("top3", [[("", 0)]])[0][0] if pred_result.get("top3") else ("", 0)
        
        with col_metrics1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Прогноз класса</div>
                <div class="metric-value">{top_pred[0]}</div>
                <div class="metric-delta delta-positive">Вероятность: {top_pred[1]*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

    # Metric 2: Probability/Confidence
    if pred_result["task"] == "classification":
        proba = pred_result.get("proba", [0])[0] if pred_result.get("proba") else 0
        threshold = pred_result.get("best_thr", 0.5)
        
        with col_metrics2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Вероятность осложнений</div>
                <div class="metric-value">{proba*100:.1f}%</div>
                <div class="metric-delta {'delta-negative' if proba >= threshold else 'delta-positive'}>
                    {'выше' if proba >= threshold else 'ниже'} порога ({threshold*100:.0f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        with col_metrics2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Тип задачи</div>
                <div class="metric-value">{pred_result["task"]}</div>
                <div class="metric-delta delta-positive">Модель обучена</div>
            </div>
            """, unsafe_allow_html=True)

    # Metric 3: Model quality
    quality_metric = "r2" if pred_result["task"] == "regression" else "accuracy"
    quality_value = metrics.get(quality_metric, metrics.get("f1", 0))
    quality_label = "R²" if quality_metric == "r2" else "F1" if quality_metric == "f1" else "Accuracy"
    
    with col_metrics3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Качество модели</div>
            <div class="metric-value">{quality_value:.2f}</div>
            <div class="metric-delta delta-positive">{quality_label} на тестовом сплите</div>
        </div>
        """, unsafe_allow_html=True)

    # Metric 4: Additional info
    with col_metrics4:
        if pred_result["task"] == "regression":
            pred_value = pred_result["pred"][0] if pred_result["pred"] else 0
            if pred_value >= 8:
                interpretation_value = "Высокий"
                interpretation_label = "условная категория по прогнозу SOFA"
            elif pred_value >= 5:
                interpretation_value = "Средний"
                interpretation_label = "условная категория по прогнозу SOFA"
            else:
                interpretation_value = "Низкий"
                interpretation_label = "условная категория по прогнозу SOFA"
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Категория прогноза</div>
                <div class="metric-value">{interpretation_value}</div>
                <div class="metric-delta delta-positive">{interpretation_label}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Классов</div>
                <div class="metric-value">{len(pred_result.get('top3', [[('', 0)]])[0]) if pred_result.get('top3') else 1}</div>
                <div class="metric-delta delta-positive">Топ предсказаний</div>
            </div>
            """, unsafe_allow_html=True)

    # ===================== БЛОК: ПРОГНОЗ ДИНАМИКИ =====================
    st.markdown('<div class="subsection-title">Визуализация прогноза</div>', unsafe_allow_html=True)
    st.caption("Для регрессии ниже показана учебная визуализация между текущим значением и прогнозом модели, а не отдельная временная модель на 72 часа.")

    # Создаем график прогноза (только для регрессии)
    if pred_result["task"] == "regression":
        fig = go.Figure()
        
        current_sofa = patient_data.get("sofa", 0)
        pred_value = pred_result["pred"][0] if pred_result["pred"] else current_sofa
        
        # Create a simple projection (linear interpolation for visualization)
        hours = [0, 24, 48, 72]
        # Simple linear projection from current to predicted
        sofa_values = [
            current_sofa,
            current_sofa + (pred_value - current_sofa) * 0.33,
            current_sofa + (pred_value - current_sofa) * 0.67,
            pred_value
        ]

        fig.add_trace(go.Scatter(
            x=hours,
            y=sofa_values,
            mode='lines+markers',
            name=f'Прогноз {cohorts[st.session_state.selected_cohort]["target"]}',
            line=dict(color='#2a5c8a', width=4),
            marker=dict(size=10, color='#4a8bc5')
        ))

        # Add current value marker
        fig.add_trace(go.Scatter(
            x=[0],
            y=[current_sofa],
            mode='markers',
            name='Текущее значение',
            marker=dict(size=15, color='#e63946', symbol='circle')
        ))

        # Add threshold line if applicable
        if pred_value > 6 or current_sofa > 6:
            fig.add_hline(
                y=6,
                line_dash="dash",
                line_color="#e63946",
                annotation_text="Критический порог",
                annotation_position="bottom right"
            )

        fig.update_layout(
            title=f"Иллюстрация перехода к прогнозу {cohorts[st.session_state.selected_cohort]['target']}",
            xaxis_title="Условная шкала визуализации",
            yaxis_title=cohorts[st.session_state.selected_cohort]['target'],
            template="plotly_white",
            height=400,
            font=dict(size=12),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        st.plotly_chart(fig, width="stretch", config={'displayModeBar': True})
    elif pred_result["task"] == "classification":
        # Show probability bar chart
        proba = pred_result.get("proba", [0])[0] if pred_result.get("proba") else 0
        threshold = pred_result.get("best_thr", 0.5)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Вероятность"],
            y=[proba * 100],
            marker_color='#2a5c8a',
            text=[f"{proba*100:.1f}%"],
            textposition='outside'
        ))
        fig.add_hline(
            y=threshold * 100,
            line_dash="dash",
            line_color="#e63946",
            annotation_text=f"Порог ({threshold*100:.0f}%)",
            annotation_position="right"
        )
        fig.update_layout(
            title="Вероятность положительного класса",
            yaxis_title="Вероятность (%)",
            yaxis_range=[0, 100],
            height=300,
            template="plotly_white"
        )
        st.plotly_chart(fig, width="stretch", config={'displayModeBar': True})
    else:  # multiclass
        # Show top-3 predictions
        top3 = pred_result.get("top3", [[("", 0)]])[0] if pred_result.get("top3") else [("", 0)]
        
        fig = go.Figure()
        classes = [item[0] for item in top3]
        probs = [item[1] * 100 for item in top3]
        
        fig.add_trace(go.Bar(
            x=classes,
            y=probs,
            marker_color='#2a5c8a',
            text=[f"{p:.1f}%" for p in probs],
            textposition='outside'
        ))
        fig.update_layout(
            title="Топ-3 предсказания классов",
            yaxis_title="Вероятность (%)",
            yaxis_range=[0, 100],
            height=300,
            template="plotly_white"
        )
        st.plotly_chart(fig, width="stretch", config={'displayModeBar': True})

    # ===================== БЛОК: ЧТО-ЕСЛИ АНАЛИЗ =====================
    st.markdown('<div class="subsection-title">Что-если анализ</div>', unsafe_allow_html=True)

    # Вместо зеленой рамки используем карточку как для рекомендаций
    st.markdown("""
    <div class="recommendation-card">
        <div class="recommendation-title">Проверьте, как изменение параметров повлияет на прогноз</div>
    """, unsafe_allow_html=True)

    # Создаем три колонки
    col_whatif1, col_whatif2, col_whatif3 = st.columns([2, 2, 1])

    with col_whatif1:
        whatif_param = st.selectbox(
            "Параметр для изменения",
            ["Прокальцитонин", "Уровень СРБ", "Лейкоциты", "Температура"],
            label_visibility="collapsed"
        )

    with col_whatif2:
        current_value = patient_data.get("pct", 2.4) if whatif_param == "Прокальцитонин" else \
                       patient_data.get("crp", 124.0) if whatif_param == "Уровень СРБ" else \
                       patient_data.get("leukocytes", 14.2) if whatif_param == "Лейкоциты" else \
                       patient_data.get("temperature", 38.5)
        
        if whatif_param == "Температура":
            whatif_value = st.slider(
                "Новое значение",
                35.0, 42.0, float(current_value), 0.1,
                label_visibility="collapsed"
            )
        elif whatif_param == "Уровень СРБ":
            whatif_value = st.slider(
                "Новое значение",
                0.0, 500.0, float(current_value), 1.0,
                label_visibility="collapsed"
            )
        elif whatif_param == "Лейкоциты":
            whatif_value = st.slider(
                "Новое значение",
                0.0, 100.0, float(current_value), 0.1,
                label_visibility="collapsed"
            )
        else:  # Прокальцитонин
            whatif_value = st.slider(
                "Новое значение",
                0.0, 50.0, float(current_value), 0.1,
                label_visibility="collapsed"
            )

    with col_whatif3:
        st.markdown("###")
        if st.button("Пересчитать", width="stretch"):
            # Create modified patient data
            modified_data = patient_data.copy()
            if whatif_param == "Прокальцитонин":
                modified_data["pct"] = whatif_value
            elif whatif_param == "Уровень СРБ":
                modified_data["crp"] = whatif_value
            elif whatif_param == "Лейкоциты":
                modified_data["leukocytes"] = whatif_value
            elif whatif_param == "Температура":
                modified_data["temperature"] = whatif_value
            
            try:
                with st.spinner("Пересчет..."):
                    new_prediction = predict_patient(modified_data, st.session_state.selected_cohort)
                    st.session_state.patient_data = modified_data
                    set_input_buffers(modified_data)
                    reset_input_widgets()
                    st.session_state.prediction_result = new_prediction
                    st.success("Прогноз пересчитан!")
                    st.rerun()
            except Exception as e:
                st.error(f"Ошибка при пересчете: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)

    # ===================== БЛОК: КЛИНИЧЕСКОЕ ЗАКЛЮЧЕНИЕ =====================
    st.markdown('<div class="subsection-title">Клиническое заключение</div>', unsafe_allow_html=True)

    # Факторы риска в плитках (на основе введенных данных)
    st.markdown('<div class="small-title">Основные факторы риска</div>', unsafe_allow_html=True)

    factors = []
    
    # Add factors based on patient data
    if patient_data.get("pct"):
        pct_val = patient_data["pct"]
        impact = "high" if pct_val > 2.0 else "medium" if pct_val > 0.5 else "low"
        factors.append({
            "name": "Прокальцитонин",
            "value": f"{pct_val:.1f} нг/мл",
            "impact": impact,
            "level": "ВЫСОКИЙ" if impact == "high" else "СРЕДНИЙ" if impact == "medium" else "НИЗКИЙ"
        })
    
    if patient_data.get("neutrophils") and patient_data.get("lymphocytes"):
        nlr = patient_data["neutrophils"] / max(patient_data["lymphocytes"], 1)
        impact = "high" if nlr > 10 else "medium" if nlr > 5 else "low"
        factors.append({
            "name": "Соотношение нейтрофилы/лимфоциты",
            "value": f"{nlr:.2f}",
            "impact": impact,
            "level": "ВЫСОКИЙ" if impact == "high" else "СРЕДНИЙ" if impact == "medium" else "НИЗКИЙ"
        })
    
    if patient_data.get("age"):
        age_val = patient_data["age"]
        impact = "high" if age_val > 70 else "medium" if age_val > 50 else "low"
        factors.append({
            "name": "Возраст пациента",
            "value": f"{age_val} лет",
            "impact": impact,
            "level": "ВЫСОКИЙ" if impact == "high" else "СРЕДНИЙ" if impact == "medium" else "НИЗКИЙ"
        })
    
    if patient_data.get("crp"):
        crp_val = patient_data["crp"]
        impact = "high" if crp_val > 100 else "medium" if crp_val > 50 else "low"
        factors.append({
            "name": "Уровень СРБ",
            "value": f"{crp_val:.0f} мг/л",
            "impact": impact,
            "level": "ВЫСОКИЙ" if impact == "high" else "СРЕДНИЙ" if impact == "medium" else "НИЗКИЙ"
        })
    
    if patient_data.get("lymphocytes"):
        lymph_val = patient_data["lymphocytes"]
        impact = "high" if lymph_val < 10 else "medium" if lymph_val < 20 else "low"
        factors.append({
            "name": "Количество лимфоцитов",
            "value": f"{lymph_val}%",
            "impact": impact,
            "level": "ВЫСОКИЙ" if impact == "high" else "СРЕДНИЙ" if impact == "medium" else "НИЗКИЙ"
        })
    
    if patient_data.get("sofa"):
        sofa_val = patient_data["sofa"]
        impact = "high" if sofa_val >= 8 else "medium" if sofa_val >= 5 else "low"
        factors.append({
            "name": "Текущая шкала SOFA",
            "value": f"{sofa_val}",
            "impact": impact,
            "level": "ВЫСОКИЙ" if impact == "high" else "СРЕДНИЙ" if impact == "medium" else "НИЗКИЙ"
        })

    if not factors:
        st.info("Введите данные пациента для анализа факторов риска")
    else:
        for factor in factors:
            impact_class = f"impact-{factor['impact']}"
            tile_class = f"factor-tile {factor['impact']}"

            st.markdown(f"""
            <div class="{tile_class}">
                <div class="factor-name">{factor['name']}</div>
                <div class="factor-value">Значение: {factor['value']}</div>
                <div class="factor-impact {impact_class}">Уровень влияния: {factor['level']}</div>
            </div>
            """, unsafe_allow_html=True)

    # Рекомендации
    st.markdown('<div class="subsection-title" style="margin-top: 2.5rem;">Учебные рекомендации</div>', unsafe_allow_html=True)
    st.warning("Ниже приведен общий учебный шаблон рекомендаций. Это не персонализированное медицинское назначение и не заменяет решение врача.")

    col_rec1, col_rec2 = st.columns(2)

    with col_rec1:
        st.markdown("""
        <div class="recommendation-card">
            <div class="recommendation-title">Мониторинг и наблюдение</div>
            <ul style="margin: 0; padding-left: 1.2rem; color: inherit;">
                <li>Усилить мониторинг витальных функций (каждые 6 часов)</li>
                <li>Контроль газового состава крови каждые 12 часов</li>
                <li>Ежедневный контроль показателей функции почек</li>
                <li>Мониторинг уровня лактата каждые 8 часов</li>
            </ul>
        </div>

        <div class="recommendation-card">
            <div class="recommendation-title">Диагностические мероприятия</div>
            <ul style="margin: 0; padding-left: 1.2rem; color: inherit;">
                <li>Повторный анализ крови с лейкоцитарной формулой через 24 часа</li>
                <li>Контрольный анализ на прокальцитонин через 48 часов</li>
                <li>УЗИ брюшной полости при сохранении симптоматики</li>
                <li>Рентгенография органов грудной клетки</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_rec2:
        st.markdown("""
        <div class="recommendation-card">
            <div class="recommendation-title">Терапевтические рекомендации</div>
            <ul style="margin: 0; padding-left: 1.2rem; color: inherit;">
                <li>Рассмотреть раннее начало антибиотикотерапии широкого спектра</li>
                <li>Провести цитокиновый профиль для оценки риска цитокинового шторма</li>
                <li>Консультация реаниматолога для решения вопроса о переводе в ОРИТ</li>
                <li>Коррекция водно-электролитного баланса</li>
            </ul>
        </div>

        <div class="recommendation-card">
            <div class="recommendation-title">Прогноз и дальнейшие действия</div>
            <ul style="margin: 0; padding-left: 1.2rem; color: inherit;">
                <li>Критический период: следующие 24-48 часов</li>
                <li>Ожидаемое время улучшения: 72-96 часов при адекватной терапии</li>
                <li>Плановый осмотр через 24 часа</li>
                <li>Повторный анализ через 48 часов</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Действия с отчетом
    st.markdown("### Действия с отчетом")
    col_actions1, col_actions2, col_actions3, col_actions4 = st.columns(4)

    with col_actions1:
        if st.button("Сохранить", width="stretch"):
            st.success("Отчет сохранен в истории пациента")

    with col_actions2:
        # Generate PDF report
        if REPORTLAB_AVAILABLE:
            pdf_buffer = generate_pdf_report(
                st.session_state.patient_data,
                st.session_state.prediction_result,
                st.session_state.selected_cohort,
                cohorts[st.session_state.selected_cohort],
                st.session_state.current_user.get("full_name", "Врач") if st.session_state.current_user else "Врач"
            )
            
            if pdf_buffer:
                pdf_filename = f"report_{st.session_state.patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                st.download_button(
                    label="Экспорт в PDF",
                    data=pdf_buffer.getvalue(),
                    file_name=pdf_filename,
                    mime="application/pdf",
                    width="stretch"
                )
            else:
                st.error("Ошибка при генерации PDF")
        else:
            st.warning("Библиотека reportlab не установлена. Установите: pip install reportlab")

    with col_actions3:
        if st.button("Новый анализ", width="stretch"):
            st.session_state.prediction_made = False
            st.rerun()

    with col_actions4:
        if st.button("В историю", width="stretch"):
            st.session_state.show_history = True
            st.rerun()

# ===================== ФУТЕР =====================
st.divider()
st.markdown("""
<div style="text-align: center; color: #aeb7c2; font-size: 0.9rem; padding: 1rem 0;">
    <p>Immunorisk Studio v1.0 • Система интеллектуального моделирования иммунного ответа</p>
    <p>© 2026 Immunorisk Research Group • Отчет по прогрессу от 21.01.2026</p>
</div>
""", unsafe_allow_html=True)
