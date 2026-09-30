"""
risk_engine.py
Rule-based chronic disease risk scoring engine.
Produces a 0-100 composite risk score and per-disease breakdown
before the LLM is invoked for narrative analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class HealthProfile:
    # Demographics
    age: int
    sex: str                            # "Male" / "Female" / "Other"
    ethnicity: str

    # Biometrics
    bmi: float
    systolic_bp: int
    diastolic_bp: int
    fasting_glucose: float              # mg/dL  (0 = not known)
    total_cholesterol: float            # mg/dL  (0 = not known)
    hdl_cholesterol: float              # mg/dL  (0 = not known)
    resting_heart_rate: int

    # Lifestyle
    smoking_status: str                 # "Never" / "Former" / "Current"
    alcohol_units_per_week: int
    physical_activity: str              # "Sedentary" / "Light" / "Moderate" / "Active"
    diet_quality: str                   # "Poor" / "Average" / "Good"
    sleep_hours: float
    stress_level: str                   # "Low" / "Moderate" / "High" / "Very High"

    # Medical history
    family_history: List[str] = field(default_factory=list)
    existing_conditions: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)


@dataclass
class RiskResult:
    composite_score: float              # 0-100
    risk_level: str                     # Low / Moderate / High / Very High
    disease_scores: Dict[str, float]    # per-disease 0-100
    top_risk_factors: List[str]
    protective_factors: List[str]


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def _bmi_score(bmi: float) -> float:
    if bmi < 18.5: return 20
    if bmi < 25:   return 0
    if bmi < 30:   return 25
    if bmi < 35:   return 50
    return 75


def _bp_score(systolic: int, diastolic: int) -> float:
    if systolic < 120 and diastolic < 80:  return 0
    if systolic < 130 and diastolic < 80:  return 15
    if systolic < 140 or diastolic < 90:   return 35
    if systolic < 160 or diastolic < 100:  return 60
    return 80


def _glucose_score(glucose: float) -> float:
    if glucose <= 0:   return 0     # unknown — do not penalise
    if glucose < 100:  return 0
    if glucose < 126:  return 40    # pre-diabetic
    return 75                        # diabetic range


def _cholesterol_score(total: float, hdl: float) -> float:
    if total <= 0:
        return 0
    score = 0.0
    if total >= 240:
        score += 40
    elif total >= 200:
        score += 20
    if hdl > 0:
        if hdl < 40:
            score += 30
        elif hdl < 60:
            score += 10
    return _clamp(score)


def _smoking_score(status: str) -> float:
    return {"Never": 0, "Former": 20, "Current": 55}.get(status, 0)


def _activity_score(level: str) -> float:
    return {"Sedentary": 45, "Light": 25, "Moderate": 5, "Active": 0}.get(level, 20)


def _diet_score(quality: str) -> float:
    return {"Poor": 40, "Average": 15, "Good": 0}.get(quality, 15)


def _sleep_score(hours: float) -> float:
    if 7 <= hours <= 9:       return 0
    if 6 <= hours < 7 or 9 < hours <= 10:  return 15
    return 35


def _stress_score(level: str) -> float:
    return {"Low": 0, "Moderate": 15, "High": 35, "Very High": 55}.get(level, 15)


def _alcohol_score(units: int) -> float:
    if units <= 7:   return 0
    if units <= 14:  return 15
    if units <= 21:  return 30
    return 50


def _age_multiplier(age: int) -> float:
    if age < 30:  return 0.7
    if age < 45:  return 0.9
    if age < 60:  return 1.1
    return 1.3


def _sex_cvd_multiplier(sex: str) -> float:
    """Women have lower CVD risk pre-menopause, higher post-menopause.
    We use age in the main scorer; here sex provides a small modifier."""
    return 1.0  # neutral — age multiplier already handles age-related risk


# ---------------------------------------------------------------------------
# Per-disease risk calculators  (weights tuned to clinical heuristics)
# ---------------------------------------------------------------------------

def _score_type2_diabetes(p: HealthProfile) -> float:
    s = 0.0
    s += _bmi_score(p.bmi) * 0.30
    s += _glucose_score(p.fasting_glucose) * 0.30
    s += _activity_score(p.physical_activity) * 0.15
    s += _diet_score(p.diet_quality) * 0.10
    s += _stress_score(p.stress_level) * 0.05
    s += 30 if "Diabetes" in p.family_history else 0
    s += 20 if "Type 2 Diabetes" in p.existing_conditions else 0
    # South-Asian ethnicity carries ~3× higher T2D risk at lower BMIs
    if "South Asian" in p.ethnicity and p.bmi >= 23:
        s += 10
    return _clamp(s * _age_multiplier(p.age))


def _score_cardiovascular(p: HealthProfile) -> float:
    s = 0.0
    s += _bp_score(p.systolic_bp, p.diastolic_bp) * 0.25
    s += _cholesterol_score(p.total_cholesterol, p.hdl_cholesterol) * 0.20
    s += _smoking_score(p.smoking_status) * 0.20
    s += _bmi_score(p.bmi) * 0.10
    s += _activity_score(p.physical_activity) * 0.10
    s += _diet_score(p.diet_quality) * 0.05
    s += 30 if "Heart Disease" in p.family_history else 0
    s += 15 if "Stroke" in p.family_history else 0
    s += 10 if "Cardiovascular Disease" in p.existing_conditions else 0
    # Resting HR > 100 is an independent risk factor
    if p.resting_heart_rate > 100:
        s += 10
    elif p.resting_heart_rate < 60:
        pass  # bradycardia in active people is protective — no penalty
    return _clamp(s * _age_multiplier(p.age))


def _score_hypertension(p: HealthProfile) -> float:
    s = 0.0
    s += _bp_score(p.systolic_bp, p.diastolic_bp) * 0.40
    s += _bmi_score(p.bmi) * 0.15
    s += _stress_score(p.stress_level) * 0.15
    s += _alcohol_score(p.alcohol_units_per_week) * 0.10
    s += _activity_score(p.physical_activity) * 0.10
    s += 25 if "Hypertension" in p.family_history else 0
    s += 15 if "Hypertension" in p.existing_conditions else 0
    return _clamp(s * _age_multiplier(p.age))


def _score_obesity_metabolic(p: HealthProfile) -> float:
    s = 0.0
    s += _bmi_score(p.bmi) * 0.40
    s += _activity_score(p.physical_activity) * 0.20
    s += _diet_score(p.diet_quality) * 0.20
    s += _sleep_score(p.sleep_hours) * 0.10
    s += _stress_score(p.stress_level) * 0.05
    s += 20 if "Obesity" in p.family_history else 0
    return _clamp(s)


def _score_mental_health(p: HealthProfile) -> float:
    s = 0.0
    s += _stress_score(p.stress_level) * 0.35
    s += _sleep_score(p.sleep_hours) * 0.25
    s += _activity_score(p.physical_activity) * 0.15
    s += _alcohol_score(p.alcohol_units_per_week) * 0.10
    s += _smoking_score(p.smoking_status) * 0.05
    s += 25 if "Mental Health" in p.family_history else 0
    return _clamp(s)


def _score_chronic_kidney(p: HealthProfile) -> float:
    s = 0.0
    s += _bp_score(p.systolic_bp, p.diastolic_bp) * 0.30
    s += _glucose_score(p.fasting_glucose) * 0.25
    s += _bmi_score(p.bmi) * 0.15
    s += 30 if "Kidney Disease" in p.family_history else 0
    s += 15 if "Type 2 Diabetes" in p.existing_conditions else 0
    s += 15 if "Hypertension" in p.existing_conditions else 0
    return _clamp(s * _age_multiplier(p.age))


# ---------------------------------------------------------------------------
# Composite scorer
# ---------------------------------------------------------------------------

DISEASE_WEIGHTS = {
    "Type 2 Diabetes":        0.20,
    "Cardiovascular Disease": 0.25,
    "Hypertension":           0.20,
    "Obesity / Metabolic":    0.15,
    "Mental Health":          0.10,
    "Chronic Kidney Disease": 0.10,
}

DISEASE_SCORERS = {
    "Type 2 Diabetes":        _score_type2_diabetes,
    "Cardiovascular Disease": _score_cardiovascular,
    "Hypertension":           _score_hypertension,
    "Obesity / Metabolic":    _score_obesity_metabolic,
    "Mental Health":          _score_mental_health,
    "Chronic Kidney Disease": _score_chronic_kidney,
}


def _risk_level(score: float) -> str:
    if score < 25:  return "Low"
    if score < 50:  return "Moderate"
    if score < 72:  return "High"
    return "Very High"


def _identify_factors(p: HealthProfile) -> tuple[List[str], List[str]]:
    risks: List[str] = []
    protective: List[str] = []

    # BMI
    if p.bmi >= 30:
        risks.append(f"Obesity (BMI {p.bmi:.1f})")
    elif p.bmi >= 25:
        risks.append(f"Overweight (BMI {p.bmi:.1f})")
    elif p.bmi < 18.5:
        risks.append(f"Underweight (BMI {p.bmi:.1f})")
    else:
        protective.append(f"Healthy BMI ({p.bmi:.1f})")

    # Blood pressure
    if p.systolic_bp >= 140 or p.diastolic_bp >= 90:
        risks.append(f"High blood pressure ({p.systolic_bp}/{p.diastolic_bp} mmHg)")
    elif p.systolic_bp >= 130:
        risks.append(f"Elevated blood pressure ({p.systolic_bp}/{p.diastolic_bp} mmHg)")
    elif p.systolic_bp < 120 and p.diastolic_bp < 80:
        protective.append("Normal blood pressure")

    # Smoking
    if p.smoking_status == "Current":
        risks.append("Current smoker")
    elif p.smoking_status == "Former":
        risks.append("Former smoker")
    elif p.smoking_status == "Never":
        protective.append("Non-smoker")

    # Physical activity
    if p.physical_activity == "Sedentary":
        risks.append("Sedentary lifestyle")
    elif p.physical_activity in ("Moderate", "Active"):
        protective.append(f"{p.physical_activity} physical activity")

    # Diet
    if p.diet_quality == "Poor":
        risks.append("Poor diet quality")
    elif p.diet_quality == "Good":
        protective.append("Good diet quality")

    # Stress
    if p.stress_level in ("High", "Very High"):
        risks.append(f"{p.stress_level} stress level")
    elif p.stress_level == "Low":
        protective.append("Low stress level")

    # Sleep
    if not (7 <= p.sleep_hours <= 9):
        risks.append(f"Suboptimal sleep ({p.sleep_hours}h/night)")
    else:
        protective.append(f"Adequate sleep ({p.sleep_hours}h/night)")

    # Alcohol
    if p.alcohol_units_per_week > 14:
        risks.append(f"High alcohol intake ({p.alcohol_units_per_week} units/week)")

    # Lab values
    if p.fasting_glucose >= 100:
        label = "Pre-diabetic" if p.fasting_glucose < 126 else "Diabetic-range"
        risks.append(f"{label} fasting glucose ({p.fasting_glucose:.0f} mg/dL)")

    if p.total_cholesterol >= 200:
        risks.append(f"Elevated total cholesterol ({p.total_cholesterol:.0f} mg/dL)")
    elif p.total_cholesterol > 0:
        protective.append(f"Desirable cholesterol ({p.total_cholesterol:.0f} mg/dL)")

    if 0 < p.hdl_cholesterol < 40:
        risks.append(f"Low HDL cholesterol ({p.hdl_cholesterol:.0f} mg/dL)")
    elif p.hdl_cholesterol >= 60:
        protective.append(f"High HDL (good) cholesterol ({p.hdl_cholesterol:.0f} mg/dL)")

    # Family history
    if p.family_history:
        risks.append(f"Family history: {', '.join(p.family_history)}")

    return risks[:7], protective[:5]


def compute_risk(profile: HealthProfile) -> RiskResult:
    disease_scores = {d: scorer(profile) for d, scorer in DISEASE_SCORERS.items()}
    composite = sum(disease_scores[d] * DISEASE_WEIGHTS[d] for d in DISEASE_WEIGHTS)
    composite = _clamp(composite)
    risk_factors, protective = _identify_factors(profile)
    return RiskResult(
        composite_score=round(composite, 1),
        risk_level=_risk_level(composite),
        disease_scores={d: round(v, 1) for d, v in disease_scores.items()},
        top_risk_factors=risk_factors,
        protective_factors=protective,
    )
