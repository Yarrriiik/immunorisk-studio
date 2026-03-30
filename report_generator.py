"""
Report Generator - Generates PDF reports for patient analysis
"""
from io import BytesIO
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


def generate_pdf_report(
    patient_data: Dict[str, Any],
    prediction_result: Dict[str, Any],
    cohort: str,
    cohort_info: Dict[str, Any],
    doctor_name: str
) -> Optional[BytesIO]:
    """
    Generate PDF report for patient analysis
    
    Returns:
        BytesIO object with PDF content, or None if reportlab is not available
    """
    if not REPORTLAB_AVAILABLE:
        return None
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#2a5c8a'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2a5c8a'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Title (English only to avoid font issues)
    story.append(Paragraph("Immunorisk Studio", title_style))
    story.append(Paragraph("Patient Analysis Report", styles['Heading2']))
    story.append(Spacer(1, 0.5*cm))
    
    # Patient Information
    story.append(Paragraph("Patient Information", heading_style))
    patient_info_data = [
        ["Parameter", "Value"],
        ["Patient ID", patient_data.get("patient_id", "N/A")],
        ["Analysis date", datetime.now().strftime("%d.%m.%Y %H:%M")],
        ["Cohort", cohort],
        ["Doctor", doctor_name],
    ]
    
    if "age" in patient_data:
        patient_info_data.append(["Age", str(patient_data.get("age", "N/A"))])
    if "sex" in patient_data:
        # Map Russian sex to English for report
        sex_raw = str(patient_data.get("sex", "N/A"))
        if sex_raw in ("Мужской", "М"):
            sex_val = "Male"
        elif sex_raw in ("Женский", "Ж"):
            sex_val = "Female"
        else:
            sex_val = sex_raw
        patient_info_data.append(["Sex", sex_val])
    
    patient_table = Table(patient_info_data, colWidths=[6*cm, 10*cm])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2a5c8a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 0.5*cm))
    
    # Prediction Results
    story.append(Paragraph("Prediction Results", heading_style))
    
    # Calculate risk level
    risk_level = "Low"
    if prediction_result["task"] == "regression":
        pred_value = prediction_result["pred"][0] if prediction_result["pred"] else 0
        if pred_value >= 8:
            risk_level = "High"
        elif pred_value >= 5:
            risk_level = "Medium"
    elif prediction_result["task"] == "classification":
        proba = prediction_result.get("proba", [0])[0] if prediction_result.get("proba") else 0
        if proba >= 0.7:
            risk_level = "High"
        elif proba >= 0.4:
            risk_level = "Medium"
    
    results_data = [
        ["Parameter", "Value"],
        ["Task type", prediction_result["task"]],
        ["Risk level", risk_level],
    ]
    
    if prediction_result["task"] == "regression":
        pred_value = prediction_result["pred"][0] if prediction_result["pred"] else 0
        results_data.append(["Prediction", f"{pred_value:.2f}"])
        results_data.append(["Target metric", cohort_info.get("target", "N/A")])
    elif prediction_result["task"] == "classification":
        proba = prediction_result.get("proba", [0])[0] if prediction_result.get("proba") else 0
        pred_binary = prediction_result.get("pred", [0])[0] if prediction_result.get("pred") else 0
        results_data.append(["Probability", f"{proba*100:.1f}%"])
        results_data.append(["Prediction", "Positive" if pred_binary == 1 else "Negative"])
        threshold = prediction_result.get("best_thr", 0.5)
        results_data.append(["Threshold", f"{threshold*100:.0f}%"])
    else:  # multiclass
        top_pred = prediction_result.get("top3", [[("", 0)]])[0][0] if prediction_result.get("top3") else ("", 0)
        results_data.append(["Predicted class", top_pred[0]])
        results_data.append(["Probability", f"{top_pred[1]*100:.1f}%"])
    
    results_table = Table(results_data, colWidths=[6*cm, 10*cm])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3a9e7a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
    ]))
    story.append(results_table)
    story.append(Spacer(1, 0.5*cm))
    
    # Key Patient Data
    story.append(Paragraph("Key patient metrics", heading_style))
    key_data = []
    key_fields = ["leukocytes", "crp", "pct", "sofa", "platelets", "creatinine", "bilirubin", 
                  "neutrophils", "lymphocytes", "temperature"]
    
    for field in key_fields:
        if field in patient_data:
            value = patient_data[field]
            field_name = {
                "leukocytes": "Leukocytes",
                "crp": "C-reactive protein (CRP)",
                "pct": "Procalcitonin",
                "sofa": "SOFA score",
                "platelets": "Platelets",
                "creatinine": "Creatinine",
                "bilirubin": "Bilirubin",
                "neutrophils": "Neutrophils",
                "lymphocytes": "Lymphocytes",
                "temperature": "Temperature"
            }.get(field, field)
            key_data.append([field_name, str(value)])
    
    if key_data:
        key_data.insert(0, ["Metric", "Value"])
        key_table = Table(key_data, colWidths=[6*cm, 10*cm])
        key_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a8bc5')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
        ]))
        story.append(key_table)
        story.append(Spacer(1, 0.5*cm))
    
    # Footer
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph(
        f"Report generated: {datetime.now().strftime('%d.%m.%Y %H:%M')}",
        ParagraphStyle('Footer', parent=styles['Normal'], fontSize=9, alignment=TA_CENTER, textColor=colors.grey)
    ))
    story.append(Paragraph(
        "Immunorisk Studio v1.0 • Intelligent immune response modelling system",
        ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, alignment=TA_CENTER, textColor=colors.grey)
    ))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer


def generate_csv_history(history_data: list[Dict[str, Any]]) -> bytes:
    """
    Generate CSV bytes from history data for download
    
    Returns:
        CSV bytes (utf-8-sig encoded for Excel compatibility)
    """
    if not history_data:
        # Return empty CSV with headers
        headers = "ID пациента,Дата анализа,Когорта,Уровень риска,SOFA,Врач,Статус\n"
        return headers.encode('utf-8-sig')
    
    df = pd.DataFrame(history_data)
    
    # Map column names - handle both "id" and "patient_id"
    if "id" in df.columns and "patient_id" not in df.columns:
        df["patient_id"] = df["id"]
    elif "patient_id" not in df.columns:
        # If neither exists, try to create from available data
        df["patient_id"] = df.get("id", "N/A")
    
    # Ensure we have the right columns
    csv_columns = ["patient_id", "date", "cohort", "risk", "sofa", "doctor", "status"]
    
    # Select only available columns
    available_columns = [col for col in csv_columns if col in df.columns]
    if not available_columns:
        # Fallback: use all available columns
        available_columns = [col for col in df.columns if col not in ["prediction", "id"]]
    
    df_export = df[available_columns].copy()
    
    # Rename columns for better readability
    column_mapping = {
        "patient_id": "ID пациента",
        "date": "Дата анализа",
        "cohort": "Когорта",
        "risk": "Уровень риска",
        "sofa": "SOFA",
        "doctor": "Врач",
        "status": "Статус"
    }
    df_export = df_export.rename(columns=column_mapping)
    
    # Convert to CSV bytes
    csv_string = df_export.to_csv(index=False, encoding='utf-8-sig')
    return csv_string.encode('utf-8-sig')
