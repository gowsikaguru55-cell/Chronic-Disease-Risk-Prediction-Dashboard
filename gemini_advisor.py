"""
gemini_advisor.py
Handles all Gemini 3.6 Flash interactions for the health risk dashboard.
Uses the new google-genai SDK (google-genai>=1.0).
"""

import os
import textwrap
from typing import Optional

from google import genai
from google.genai import types
from dotenv import load_dotenv

from risk_engine import HealthProfile, RiskResult

load_dotenv()

# Ordered list of models to try — falls back if one is unavailable for this key/region.
# gemini-3.6-flash is the latest; gemini-2.0-flash is the stable fallback.
_MODELS_PREFERENCE = [
    "gemini-3.6-flash",
    "gemini-3.6-flash-latest",
    "gemini-2.0-flash",
    "gemini-2.0-flash-latest",
    "gemini-1.5-flash",
]


def _get_client(api_key: Optional[str] = None) -> genai.Client:
    """Return a configured Gemini client (v1beta, default Google AI endpoint)."""
    key = api_key or os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise ValueError(
            "Gemini API key is not set. Enter it in the sidebar or add it to your .env file."
        )
    return genai.Client(api_key=key)


def _resolve_model(client: genai.Client) -> str:
    """Return the first model from _MODELS_PREFERENCE that is available."""
    try:
        available = {m.name.split("/")[-1] for m in client.models.list()}
        for model in _MODELS_PREFERENCE:
            if model in available:
                return model
    except Exception:
        pass  # Can't list models — just try in order at call time
    return _MODELS_PREFERENCE[0]  # will surface a clear error if truly unavailable


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_assessment_prompt(profile: HealthProfile, result: RiskResult) -> str:
    fh = ", ".join(profile.family_history) if profile.family_history else "None reported"
    ec = ", ".join(profile.existing_conditions) if profile.existing_conditions else "None"
    meds = ", ".join(profile.medications) if profile.medications else "None"

    disease_breakdown = "\n".join(
        f"  • {disease}: {score}/100"
        for disease, score in sorted(result.disease_scores.items(), key=lambda x: -x[1])
    )
    risk_factors = "\n".join(f"  • {f}" for f in result.top_risk_factors) or "  • None identified"
    protective = "\n".join(f"  • {f}" for f in result.protective_factors) or "  • None identified"

    glucose_str = (
        f"{profile.fasting_glucose} mg/dL" if profile.fasting_glucose > 0 else "Not measured"
    )
    chol_str = (
        f"{profile.total_cholesterol} mg/dL" if profile.total_cholesterol > 0 else "Not measured"
    )
    hdl_str = (
        f"{profile.hdl_cholesterol} mg/dL" if profile.hdl_cholesterol > 0 else "Not measured"
    )

    return textwrap.dedent(f"""
    You are a highly knowledgeable preventive medicine physician and health coach.
    Analyse the following patient health profile and risk assessment, then provide
    a comprehensive, empathetic, and actionable health report.

    ═══════════════════════════════════════════════════
    PATIENT HEALTH PROFILE
    ═══════════════════════════════════════════════════
    Demographics  : Age {profile.age}, {profile.sex}, Ethnicity: {profile.ethnicity}
    Biometrics    : BMI {profile.bmi:.1f} | BP {profile.systolic_bp}/{profile.diastolic_bp} mmHg
                    Fasting Glucose: {glucose_str}
                    Total Cholesterol: {chol_str}
                    HDL Cholesterol: {hdl_str}
                    Resting Heart Rate: {profile.resting_heart_rate} bpm
    Lifestyle     : Smoking: {profile.smoking_status} | Alcohol: {profile.alcohol_units_per_week} units/week
                    Activity: {profile.physical_activity} | Diet: {profile.diet_quality}
                    Sleep: {profile.sleep_hours}h/night | Stress: {profile.stress_level}
    Medical History: Family History — {fh}
                     Existing Conditions — {ec}
                     Medications — {meds}

    ═══════════════════════════════════════════════════
    COMPUTED RISK SCORES  (0 = minimal risk, 100 = very high risk)
    ═══════════════════════════════════════════════════
    Overall Composite Risk Score : {result.composite_score}/100 ({result.risk_level} Risk)

    Disease-Specific Breakdown:
{disease_breakdown}

    Key Risk Factors Identified:
{risk_factors}

    Protective Factors:
{protective}

    ═══════════════════════════════════════════════════
    REQUIRED REPORT STRUCTURE (use these exact headings)
    ═══════════════════════════════════════════════════

    ## 🩺 Health Risk Summary
    Write a concise 2-3 sentence personalised overview of the patient's overall risk
    profile, mentioning the composite score and primary concerns.

    ## ⚠️ Top Health Risks & Why
    For each of the top 3 disease categories by score, explain IN PLAIN LANGUAGE:
    - What the risk score means for this specific patient
    - Which personal factors are driving that risk
    - What could happen if not addressed

    ## ✅ Preventive Action Plan
    Provide a structured, prioritised action plan with:
    - **Immediate Actions (next 30 days)**: 3-4 specific, measurable steps
    - **Short-term Goals (1-3 months)**: 3-4 lifestyle modifications with targets
    - **Long-term Habits (3-12 months)**: 3-4 sustainable behaviour changes

    ## 🥗 Personalised Nutrition Guidance
    Give specific dietary recommendations tailored to the patient's top risk factors.
    Include foods to prioritise, foods to limit, and a simple daily meal pattern.

    ## 🏃 Exercise Prescription
    Based on current activity level ({profile.physical_activity}), prescribe:
    - Weekly exercise type, frequency, duration, and intensity
    - How to safely progress over the next 3 months

    ## 🧠 Mental & Emotional Wellbeing
    Address stress ({profile.stress_level}) and sleep ({profile.sleep_hours}h) with
    evidence-based strategies and specific techniques.

    ## 🔬 Recommended Health Screenings
    List specific tests and check-ups this patient should schedule, with frequency.

    ## 💡 Motivational Summary
    Close with a 2-3 sentence personalised encouragement that acknowledges their
    strengths and empowers them to take action.

    ─────────────────────────────────────────────────────────────────────
    IMPORTANT GUIDELINES:
    - Use plain, warm, encouraging language — avoid medical jargon
    - Be specific and actionable, not generic
    - Acknowledge any protective factors the patient already has
    - Do NOT diagnose; recommend professional consultation where appropriate
    - Format using markdown with clear headers, bullet points, and bold key terms
    ─────────────────────────────────────────────────────────────────────
    """).strip()


def _build_chat_system_prompt(profile: HealthProfile, result: RiskResult) -> str:
    fh = ", ".join(profile.family_history) if profile.family_history else "None"
    top3 = ", ".join(
        sorted(result.disease_scores, key=result.disease_scores.__getitem__, reverse=True)[:3]
    )
    return textwrap.dedent(f"""
    You are a helpful preventive health assistant with access to this patient's profile:
    - Age: {profile.age}, Sex: {profile.sex}, BMI: {profile.bmi:.1f}
    - BP: {profile.systolic_bp}/{profile.diastolic_bp} mmHg | Heart rate: {profile.resting_heart_rate} bpm
    - Smoking: {profile.smoking_status} | Alcohol: {profile.alcohol_units_per_week} units/week
    - Activity: {profile.physical_activity} | Diet: {profile.diet_quality}
    - Stress: {profile.stress_level} | Sleep: {profile.sleep_hours}h/night
    - Family history: {fh}
    - Overall Risk Score: {result.composite_score}/100 ({result.risk_level})
    - Top 3 disease risks: {top3}
    - Key risk factors: {', '.join(result.top_risk_factors) or 'None identified'}
    - Protective factors: {', '.join(result.protective_factors) or 'None identified'}

    Answer the user's health question in the context of their specific profile.
    Be concise, evidence-based, warm, and practical. Remind them to consult their
    doctor for clinical decisions. Format answers with markdown where helpful.
    Never diagnose. Never prescribe medications.
    """).strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_health_report(
    profile: HealthProfile,
    result: RiskResult,
    api_key: Optional[str] = None,
) -> str:
    """Generate a full AI health report using the best available Gemini Flash model."""
    client = _get_client(api_key)
    model = _resolve_model(client)
    prompt = _build_assessment_prompt(profile, result)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.4,
            top_p=0.9,
            max_output_tokens=3000,
        ),
    )
    return response.text


def ask_health_question(
    question: str,
    profile: HealthProfile,
    result: RiskResult,
    chat_history: list,
    api_key: Optional[str] = None,
) -> str:
    """
    Answer a follow-up health question using multi-turn chat.
    chat_history is a list of {"role": "user"|"assistant", "content": "..."} dicts.
    """
    client = _get_client(api_key)
    model = _resolve_model(client)
    system_prompt = _build_chat_system_prompt(profile, result)

    # Build contents list: conversation history + new question
    contents: list[types.Content] = []
    for msg in chat_history[:-1]:  # exclude the current question (already last in history)
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))
    # Add the current question
    contents.append(types.Content(role="user", parts=[types.Part(text=question)]))

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.5,
            top_p=0.9,
            max_output_tokens=1024,
        ),
    )
    return response.text
