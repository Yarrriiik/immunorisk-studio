"""
ML Service Module - Connects GUI with ML prediction models
"""
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np

from src.predict import predict_df, load_artifacts, missing_columns_for_cohort

# Mapping from GUI cohort names (Russian) to model cohort names (English)
COHORT_MAPPING = {
    "Сепсис": "sepsis",
    "Перитонит": "peritonitis",
    "ИБС": "ihd",
    "IL-2 постковид": "il2_postcovid",
    "COVID-19": None,  # Not available in models
    "Ревматоидный артрит": None,  # Not available in models
    "Онкология": None,  # Not available in models
}

# Reverse mapping for display
REVERSE_COHORT_MAPPING = {v: k for k, v in COHORT_MAPPING.items() if v is not None}


def get_available_cohorts() -> list[str]:
    """Get list of cohorts available in the GUI that have ML models"""
    from pathlib import Path
    from src.predict import ARTIFACTS_DIR
    
    available = []
    for gui_name, model_name in COHORT_MAPPING.items():
        if model_name is None:
            continue
        # Check if model file exists
        model_path = ARTIFACTS_DIR / model_name / "model.cbm"
        meta_path = ARTIFACTS_DIR / model_name / "meta.json"
        if model_path.exists() and meta_path.exists():
            available.append(gui_name)
    
    return available


def get_model_cohort_name(gui_cohort: str) -> Optional[str]:
    """Convert GUI cohort name to model cohort name"""
    return COHORT_MAPPING.get(gui_cohort)


def _normalize_numeric_value(value: Any) -> float:
    if value is None:
        return np.nan
    if isinstance(value, str):
        value = value.strip().replace(",", ".")
        if not value:
            return np.nan
    return float(value)


def _normalize_categorical_value(feature: str, value: Any) -> str:
    value_str = str(value).strip()
    if feature.lower() in {"sex", "пол", "пол.1"}:
        if value_str in {"Мужской", "М", "Male", "male", "m"}:
            return "М"
        if value_str in {"Женский", "Ж", "Female", "female", "f"}:
            return "Ж"
    return value_str


def _set_feature_value(
    df: pd.DataFrame,
    feature: str,
    value: Any,
    cat_features_set: set[str],
    *,
    overwrite: bool = True,
) -> bool:
    if feature not in df.columns:
        return False

    current_value = df.loc[0, feature]
    if not overwrite and pd.notna(current_value):
        return False

    if feature in cat_features_set:
        if value is None or (isinstance(value, str) and not value.strip()):
            return False
        df.loc[0, feature] = _normalize_categorical_value(feature, value)
        return True

    try:
        df.loc[0, feature] = _normalize_numeric_value(value)
        return pd.notna(df.loc[0, feature])
    except (ValueError, TypeError):
        return False


def prepare_patient_data(
    patient_data: Dict[str, Any],
    cohort: str
) -> pd.DataFrame:
    """
    Prepare patient data from GUI inputs to DataFrame format expected by model.
    Maps common GUI fields to model features and fills missing features with NaN.
    """
    model_cohort = get_model_cohort_name(cohort)
    if not model_cohort:
        raise ValueError(f"Model not available for cohort: {cohort}")
    
    # Load artifacts to get required features
    artifacts = load_artifacts(model_cohort)
    
    # Start with empty DataFrame with all required features (one row)
    df = pd.DataFrame(index=[0], columns=artifacts.features)
    
    # Map common GUI inputs to model features
    # This is a simplified mapping - in production, you'd want a more comprehensive mapping
    feature_mapping = {
        # Basic demographics
        "age": ["Возраст", "возраст", "age", "Возраст.1", "лет"],
        "sex": ["sex", "Sex", "Пол", "пол", "Пол.1"],
        "glasgow": ["Шкала Глазго", "Шкала Глазго.1"],
        "urea": ["Мочевина", "Мочевина.1", "БАК мочевина"],
        "potassium": ["Калий", "Калий.1", "калий"],
        "sodium": ["Натрий", "Натрий.1", "натрий"],
        
        # Common lab values
        "leukocytes": ["WBC Лейкоциты", "Лейкоциты", "ОАК лейкоциты", "Общие лейкоциты"],
        "crp": ["С-реактивный белок (СРБ)", "С-реактивный белок (СРБ).1", "СРБ1, мг/л", "СРБ2"],
        "pct": ["Прокальцитонин"],  # May not be in all models
        "sofa": ["Шкала SOFA", "Шкала SOFA.1", "SOFA"],
        "temperature": [],  # May not be directly in model
        
        # Blood counts
        "platelets": ["PLT Тромбоциты", "PLT Тромбоциты.1", "тромбоциты"],
        "neutrophils": ["Нейтрофилы", "Нейтрофилы.1", "нейтрофилы", "Нейтрофилы, абс"],
        "lymphocytes": ["Лимфоциты", "Лимфоциты.1", "лимфоциты", "Лимфоциты, %", "Лимфоциты, абс"],
        "creatinine": ["Креатинин", "Креатинин.1", "БАК креатинин"],
        "bilirubin": ["Билирубин общий", "Билирубин общий.1"],
    }
    
    # Fill in values from patient_data
    # Separate categorical and numeric features
    cat_features_set = set(artifacts.cat_features)

    # First, copy direct feature values when the user enters exact model columns.
    for feature, value in patient_data.items():
        _set_feature_value(df, feature, value, cat_features_set)
    
    # First, handle all numeric features
    for gui_key, model_features in feature_mapping.items():
        if gui_key in patient_data and gui_key != "sex":
            value = patient_data[gui_key]
            # Try to map to available model features
            for feature in model_features:
                if feature in artifacts.features and feature not in cat_features_set:
                    # Only set numeric features
                    _set_feature_value(df, feature, value, cat_features_set, overwrite=False)
    
    # Handle sex/gender mapping - only for categorical features
    if "sex" in patient_data:
        sex_value = patient_data["sex"]
        # Map to model format
        sex_mapped = "М" if sex_value == "Мужской" else "Ж" if sex_value == "Женский" else str(sex_value)
        
        for sex_feature in ["sex", "Sex", "Пол", "пол", "Пол.1"]:
            if sex_feature in artifacts.features and sex_feature in cat_features_set:
                _set_feature_value(df, sex_feature, sex_mapped, cat_features_set, overwrite=False)
    
    # Don't do any type conversion here - prepare_features in predict.py will handle it
    # It will:
    # 1. Make unique column names (handle duplicates)
    # 2. Add missing columns as NaN
    # 3. Convert categorical features to strings with "__MISSING__" for NaN
    # 4. Keep numeric features as numeric
    
    return df


def predict_patient(
    patient_data: Dict[str, Any],
    cohort: str,
    artifacts_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Make prediction for a patient using the ML model.
    
    Args:
        patient_data: Dictionary with patient data from GUI
        cohort: GUI cohort name (Russian)
        artifacts_dir: Optional path to artifacts directory
        
    Returns:
        Dictionary with prediction results
    """
    model_cohort = get_model_cohort_name(cohort)
    if not model_cohort:
        raise ValueError(f"Model not available for cohort: {cohort}")
    
    # Prepare data
    df = prepare_patient_data(patient_data, cohort)
    
    # Make prediction
    if artifacts_dir:
        result = predict_df(model_cohort, df, artifacts_dir=artifacts_dir)
    else:
        result = predict_df(model_cohort, df)
    
    return result


def get_model_info(cohort: str) -> Optional[Dict[str, Any]]:
    """Get information about the model for a cohort"""
    model_cohort = get_model_cohort_name(cohort)
    if not model_cohort:
        return None
    
    try:
        artifacts = load_artifacts(model_cohort)
        return {
            "cohort": model_cohort,
            "task": artifacts.task,
            "target_col": artifacts.target_col,
            "n_features": len(artifacts.features),
            "metrics": artifacts.metrics,
        }
    except Exception as e:
        return None


def check_missing_features(
    patient_data: Dict[str, Any],
    cohort: str
) -> list[str]:
    """Check which required features are missing from patient data"""
    model_cohort = get_model_cohort_name(cohort)
    if not model_cohort:
        return []
    
    df = prepare_patient_data(patient_data, cohort)
    return missing_columns_for_cohort(model_cohort, df)


def get_input_coverage(patient_data: Dict[str, Any], cohort: str) -> Dict[str, Any]:
    """Return how many model features are actually populated after GUI mapping."""
    model_cohort = get_model_cohort_name(cohort)
    if not model_cohort:
        return {
            "filled_count": 0,
            "missing_count": 0,
            "total_features": 0,
            "coverage": 0.0,
            "filled_features": [],
            "missing_features": [],
        }

    artifacts = load_artifacts(model_cohort)
    df = prepare_patient_data(patient_data, cohort)
    row = df.iloc[0]
    filled_features = [feature for feature in artifacts.features if pd.notna(row[feature])]
    missing_features = [feature for feature in artifacts.features if pd.isna(row[feature])]
    total_features = len(artifacts.features)

    return {
        "filled_count": len(filled_features),
        "missing_count": len(missing_features),
        "total_features": total_features,
        "coverage": len(filled_features) / total_features if total_features else 0.0,
        "filled_features": filled_features,
        "missing_features": missing_features,
    }
