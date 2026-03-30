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
        "age": ["Возраст", "age", "Возраст.1"],
        "sex": ["sex", "Пол", "Пол.1"],
        
        # Common lab values
        "leukocytes": ["WBC Лейкоциты", "Лейкоциты"],
        "crp": ["С-реактивный белок (СРБ)", "С-реактивный белок (СРБ).1"],
        "pct": ["Прокальцитонин"],  # May not be in all models
        "sofa": ["Шкала SOFA", "Шкала SOFA.1"],
        "temperature": [],  # May not be directly in model
        
        # Blood counts
        "platelets": ["PLT Тромбоциты", "PLT Тромбоциты.1"],
        "neutrophils": ["Нейтрофилы", "Нейтрофилы.1"],
        "lymphocytes": ["Лимфоциты", "Лимфоциты.1"],
        "creatinine": ["Креатинин", "Креатинин.1"],
        "bilirubin": ["Билирубин общий", "Билирубин общий.1"],
    }
    
    # Fill in values from patient_data
    # Separate categorical and numeric features
    cat_features_set = set(artifacts.cat_features)
    
    # First, handle all numeric features
    for gui_key, model_features in feature_mapping.items():
        if gui_key in patient_data and gui_key != "sex":
            value = patient_data[gui_key]
            # Try to map to available model features
            for feature in model_features:
                if feature in artifacts.features and feature not in cat_features_set:
                    # Only set numeric features
                    try:
                        num_value = float(value) if value is not None else np.nan
                        df.loc[0, feature] = num_value
                    except (ValueError, TypeError):
                        # If can't convert, leave as NaN
                        pass
    
    # Handle sex/gender mapping - only for categorical features
    if "sex" in patient_data:
        sex_value = patient_data["sex"]
        # Map to model format
        sex_mapped = "М" if sex_value == "Мужской" else "Ж" if sex_value == "Женский" else str(sex_value)
        
        for sex_feature in ["sex", "Пол", "Пол.1"]:
            if sex_feature in artifacts.features and sex_feature in cat_features_set:
                # Only set for categorical features
                df.loc[0, sex_feature] = str(sex_mapped)
    
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
