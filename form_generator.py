"""
Dynamic Form Generator - Creates input forms based on model features
"""
import streamlit as st
from typing import Dict, Any, List, Set
from ml_service import get_model_info, get_model_cohort_name
from src.predict import load_artifacts


def get_cohort_features(cohort: str) -> Dict[str, Any]:
    """
    Get features required for a cohort
    
    Returns:
        {
            "features": list of feature names,
            "cat_features": set of categorical feature names,
            "common_features": list of common/important features for UI
        }
    """
    model_cohort = get_model_cohort_name(cohort)
    if not model_cohort:
        return {"features": [], "cat_features": set(), "common_features": []}
    
    try:
        artifacts = load_artifacts(model_cohort)
        cat_features_set = set(artifacts.cat_features)
        
        # Identify common/important features for UI (non-duplicated, meaningful names)
        common_features = []
        seen_bases = set()
        
        for feature in artifacts.features:
            # Skip duplicates (features with .1, .2, etc.)
            base_name = feature.split('.')[0]
            if base_name not in seen_bases and feature not in cat_features_set:
                # Prioritize common medical parameters
                if any(keyword in feature.lower() for keyword in [
                    'возраст', 'age', 'лейкоцит', 'leukocyte', 'wbc',
                    'тромбоцит', 'platelet', 'plt', 'гемоглобин', 'hemoglobin', 'hgb',
                    'креатинин', 'creatinine', 'мочевина', 'urea', 'глюкоза', 'glucose',
                    'срб', 'crp', 'sofa', 'глазго', 'glasgow', 'нейтрофил', 'neutrophil',
                    'лимфоцит', 'lymphocyte', 'билирубин', 'bilirubin'
                ]):
                    common_features.append(feature)
                    seen_bases.add(base_name)
                    if len(common_features) >= 20:  # Limit to 20 most important
                        break
        
        return {
            "features": artifacts.features,
            "cat_features": cat_features_set,
            "common_features": common_features
        }
    except Exception as e:
        return {"features": [], "cat_features": set(), "common_features": []}


def generate_dynamic_form(cohort: str, patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate dynamic form based on cohort features
    
    Returns:
        Updated patient_data dictionary
    """
    features_info = get_cohort_features(cohort)
    common_features = features_info["common_features"]
    cat_features = features_info["cat_features"]
    all_features = features_info["features"]
    
    if not all_features:
        st.warning("Не удалось загрузить параметры модели для этой когорты")
        return patient_data
    
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    
    # Group features by category
    basic_features = []
    lab_features = []
    immune_features = []
    other_features = []
    
    for feature in common_features:
        feature_lower = feature.lower()
        if any(x in feature_lower for x in ['возраст', 'age', 'пол', 'sex']):
            basic_features.append(feature)
        elif any(x in feature_lower for x in ['лейкоцит', 'leukocyte', 'тромбоцит', 'platelet', 
                                               'гемоглобин', 'hemoglobin', 'креатинин', 'creatinine',
                                               'мочевина', 'urea', 'глюкоза', 'glucose', 'срб', 'crp',
                                               'sofa', 'глазго', 'glasgow', 'нейтрофил', 'neutrophil',
                                               'лимфоцит', 'lymphocyte', 'билирубин', 'bilirubin']):
            lab_features.append(feature)
        elif any(x in feature_lower for x in ['cd', 'il-', 'tnf', 'ig', 'nk', 'treg', 'b-cell']):
            immune_features.append(feature)
        else:
            other_features.append(feature)
    
    # Basic information
    if basic_features:
        st.markdown("### Основная информация")
        col1, col2 = st.columns(2)
        
        with col1:
            # Age
            age_features = [f for f in basic_features if 'возраст' in f.lower() or 'age' in f.lower()]
            if age_features:
                age = st.slider("Возраст (лет)", 18, 100, 
                              value=int(patient_data.get("age", patient_data.get("Возраст", 62))))
                patient_data["age"] = age
                for af in age_features:
                    patient_data[af] = age
            
            # Sex
            sex_features = [f for f in all_features if f in cat_features and ('пол' in f.lower() or 'sex' in f.lower())]
            if sex_features:
                sex_options = ["Мужской", "Женский"]
                current_sex = patient_data.get("sex", "Мужской")
                sex_index = 0 if current_sex == "Мужской" else 1
                sex = st.selectbox("Пол", sex_options, index=sex_index)
                patient_data["sex"] = sex
        
        with col2:
            # Patient ID
            patient_id = st.text_input("ID пациента", value=patient_data.get("patient_id", "P-001"))
            patient_data["patient_id"] = patient_id
    
    # Laboratory values
    if lab_features:
        st.markdown("### Лабораторные показатели")
        
        # Group lab features into columns
        num_cols = min(3, len(lab_features))
        cols = st.columns(num_cols)
        
        for idx, feature in enumerate(lab_features[:15]):  # Limit to 15 for UI
            col_idx = idx % num_cols
            with cols[col_idx]:
                # Determine input type and range based on feature name
                feature_lower = feature.lower()
                current_value = patient_data.get(feature, 0.0)
                
                if 'sofa' in feature_lower or 'глазго' in feature_lower or 'glasgow' in feature_lower:
                    value = st.slider(feature, 0, 24, value=int(current_value) if current_value else 0)
                elif 'лейкоцит' in feature_lower or 'leukocyte' in feature_lower or 'wbc' in feature_lower:
                    value = st.number_input(feature, min_value=0.0, max_value=100.0, 
                                          value=float(current_value) if current_value else 0.0, step=0.1)
                elif 'тромбоцит' in feature_lower or 'platelet' in feature_lower or 'plt' in feature_lower:
                    value = st.number_input(feature, min_value=0.0, max_value=1000.0,
                                          value=float(current_value) if current_value else 0.0, step=1.0)
                elif 'срб' in feature_lower or 'crp' in feature_lower:
                    value = st.number_input(feature, min_value=0.0, max_value=500.0,
                                          value=float(current_value) if current_value else 0.0, step=1.0)
                elif '%' in feature or 'процент' in feature_lower:
                    value = st.slider(feature, 0, 100, value=int(current_value) if current_value else 0)
                else:
                    value = st.number_input(feature, min_value=0.0, max_value=1000.0,
                                          value=float(current_value) if current_value else 0.0, step=0.1)
                
                patient_data[feature] = value
    
    # Additional features in expander
    remaining_features = [f for f in common_features if f not in basic_features and f not in lab_features[:15]]
    if remaining_features or immune_features:
        with st.expander("Дополнительные параметры"):
            st.info(f"Всего параметров в модели: {len(all_features)}. Показаны основные параметры.")
            st.info("Остальные параметры будут заполнены значениями по умолчанию (NaN), модель обработает их автоматически.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return patient_data
