"""
app.py
Chronic Disease Risk Prediction Dashboard
AI-Powered with Gemini 3.6 Flash | Built with Streamlit
"""

import streamlit as st
import os
from dotenv import load_dotenv

from risk_engine import HealthProfile, compute_risk
from gemini_advisor import generate_health_report, ask_health_question
from charts import (
    gauge_chart, disease_bar_chart, radar_chart,
    risk_factor_donut, bmi_gauge, bp_chart,
)

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chronic Disease Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────
for key, default in {
    "risk_result": None,
    "health_profile": None,
    "ai_report": None,
    "chat_history": [],
    "assessment_done": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

RISK_BADGE = {
    "Low":       "🟢 Low Risk",
    "Moderate":  "🟡 Moderate Risk",
    "High":      "🟠 High Risk",
    "Very High": "🔴 Very High Risk",
}


def _api_key() -> str:
    """Return API key: sidebar input > .env > Streamlit secrets."""
    sidebar_key = st.session_state.get("api_key_input", "").strip()
    if sidebar_key:
        return sidebar_key
    env_key = os.getenv("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key
    # Streamlit Cloud secrets
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — API Key + quick help
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 Gemini 3.6 Flash")
    st.markdown("*AI-Powered Health Risk Predictor*")
    st.divider()

    st.markdown("### ⚙️ Configuration")
    api_key_input = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AIza...",
        help="Get your free key at https://aistudio.google.com",
        key="api_key_input",
    )
    if _api_key():
        st.success("✅ API key detected", icon="🔑")
    else:
        st.warning("Enter your API key to enable AI features.", icon="🔑")

    st.caption("Your key is never stored — sent only directly to the Gemini API.")

    st.divider()
    st.markdown("### 📋 How to Use")
    st.markdown(
        """
1. Enter your **Gemini API Key** above
2. Fill in your **health data** in each tab
3. Click **Analyse My Health Risk**
4. Review your **AI-generated report**
5. Chat with the **AI Health Advisor**
        """
    )
    st.divider()
    st.markdown("### ⚠️ Disclaimer")
    st.caption(
        "This tool is for **educational purposes only** and does **not** constitute "
        "medical advice. Always consult a qualified healthcare professional."
    )
    st.divider()
    if st.session_state.assessment_done:
        if st.button("🔄 Reset & Start Over", use_container_width=True):
            for k in list(st.session_state.keys()):
                if k != "api_key_input":
                    del st.session_state[k]
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# 🩺 Chronic Disease Risk Prediction Dashboard")
st.markdown(
    "##### Powered by **Gemini 3.6 Flash** — Enter your health data to receive a "
    "personalised AI risk assessment and actionable prevention plan."
)
st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# INPUT SECTION
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.assessment_done:
    st.markdown("## 📝 Enter Your Health Information")
    st.info(
        "Complete all sections for the most accurate assessment. "
        "Fields marked ✱ are required.",
        icon="ℹ️",
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "👤 Demographics & Biometrics",
        "🏃 Lifestyle & Habits",
        "🏥 Medical History",
        "🔬 Lab Results (Optional)",
    ])

    # ── Tab 1 : Demographics & Biometrics ────────────────────────────────────
    with tab1:
        st.markdown("### Personal Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age ✱", min_value=18, max_value=100, value=35, step=1)
        with col2:
            sex = st.selectbox("Biological Sex ✱", ["Male", "Female", "Other"])
        with col3:
            ethnicity = st.selectbox(
                "Ethnicity",
                ["Prefer not to say", "White / Caucasian", "Black / African",
                 "Hispanic / Latino", "South Asian", "East Asian",
                 "Middle Eastern", "Mixed / Other"],
            )

        st.markdown("### Body Measurements")
        col1, col2, col3 = st.columns(3)
        with col1:
            weight_kg = st.number_input(
                "Weight (kg) ✱", min_value=30.0, max_value=300.0, value=75.0, step=0.5
            )
        with col2:
            height_cm = st.number_input(
                "Height (cm) ✱", min_value=100.0, max_value=250.0, value=170.0, step=0.5
            )
        with col3:
            bmi_auto = weight_kg / ((height_cm / 100) ** 2)
            st.metric(
                "Calculated BMI", f"{bmi_auto:.1f} kg/m²",
                help="Body Mass Index calculated from weight and height.",
            )

        if bmi_auto < 18.5:
            bmi_cat, bmi_color = "Underweight", "blue"
        elif bmi_auto < 25:
            bmi_cat, bmi_color = "Normal weight ✅", "green"
        elif bmi_auto < 30:
            bmi_cat, bmi_color = "Overweight ⚠️", "orange"
        else:
            bmi_cat, bmi_color = "Obese 🚨", "red"
        st.caption(f"BMI Category: **:{bmi_color}[{bmi_cat}]**")

        st.markdown("### Blood Pressure")
        col1, col2, col3 = st.columns(3)
        with col1:
            systolic_bp = st.number_input(
                "Systolic BP (mmHg) ✱", min_value=60, max_value=260, value=120, step=1,
                help="The top number in a blood pressure reading.",
            )
        with col2:
            diastolic_bp = st.number_input(
                "Diastolic BP (mmHg) ✱", min_value=40, max_value=160, value=80, step=1,
                help="The bottom number in a blood pressure reading.",
            )
        with col3:
            resting_hr = st.number_input(
                "Resting Heart Rate (bpm)", min_value=30, max_value=200, value=72, step=1
            )

        # Live BP classification
        if systolic_bp < 120 and diastolic_bp < 80:
            st.caption("🟢 Blood Pressure: **Normal**")
        elif systolic_bp < 130 and diastolic_bp < 80:
            st.caption("🟡 Blood Pressure: **Elevated**")
        elif systolic_bp < 140 or diastolic_bp < 90:
            st.caption("🟠 Blood Pressure: **Stage 1 Hypertension** — consult your doctor")
        else:
            st.caption("🔴 Blood Pressure: **Stage 2 Hypertension** — please consult your doctor")

    # ── Tab 2 : Lifestyle ─────────────────────────────────────────────────────
    with tab2:
        st.markdown("### Physical Activity & Diet")
        col1, col2 = st.columns(2)
        with col1:
            physical_activity = st.selectbox(
                "Physical Activity Level ✱",
                ["Sedentary", "Light", "Moderate", "Active"],
                help=(
                    "**Sedentary**: Little/no exercise  \n"
                    "**Light**: 1-3 days/week  \n"
                    "**Moderate**: 3-5 days/week  \n"
                    "**Active**: 6-7 days/week or intense daily activity"
                ),
            )
        with col2:
            diet_quality = st.selectbox(
                "Overall Diet Quality ✱",
                ["Poor", "Average", "Good"],
                help=(
                    "**Poor**: Mostly processed/fast food  \n"
                    "**Average**: Mixed diet  \n"
                    "**Good**: Mostly whole foods, vegetables, lean proteins"
                ),
            )

        st.markdown("### Habits")
        col1, col2, col3 = st.columns(3)
        with col1:
            smoking_status = st.selectbox("Smoking Status ✱", ["Never", "Former", "Current"])
        with col2:
            alcohol_units = st.slider(
                "Alcohol (units/week) ✱", 0, 60, 5,
                help="1 unit ≈ 1 small glass of wine, half pint of beer, or single shot of spirits.",
            )
        with col3:
            sleep_hours = st.slider(
                "Average Sleep (hours/night) ✱", 3.0, 12.0, 7.0, step=0.5
            )

        # Contextual feedback
        if smoking_status == "Current":
            st.warning("Smoking significantly increases cardiovascular and cancer risks.", icon="🚬")
        if alcohol_units > 21:
            st.warning(
                f"{alcohol_units} units/week exceeds recommended limits (14 for women, 21 for men).",
                icon="🍺",
            )
        if not (7 <= sleep_hours <= 9):
            st.info(
                f"{sleep_hours}h of sleep is outside the recommended 7-9 hours for adults.",
                icon="😴",
            )

        st.markdown("### Stress & Mental Wellbeing")
        stress_level = st.select_slider(
            "Current Stress Level ✱",
            options=["Low", "Moderate", "High", "Very High"],
            value="Moderate",
        )
        stress_descriptions = {
            "Low":       "😌 You feel generally calm and in control.",
            "Moderate":  "😐 Occasional stress that is mostly manageable.",
            "High":      "😰 Frequent stress that impacts daily life.",
            "Very High": "😫 Persistent overwhelming stress.",
        }
        st.caption(stress_descriptions[stress_level])

    # ── Tab 3 : Medical History ───────────────────────────────────────────────
    with tab3:
        st.markdown("### Family Medical History")
        st.caption(
            "Select conditions diagnosed in your **immediate family** (parents, siblings)."
        )
        col1, col2 = st.columns(2)
        with col1:
            fh_diabetes     = st.checkbox("🩸 Diabetes (Type 1 or 2)")
            fh_heart        = st.checkbox("❤️ Heart Disease / Heart Attack")
            fh_stroke       = st.checkbox("🧠 Stroke")
            fh_hypertension = st.checkbox("🩺 High Blood Pressure")
        with col2:
            fh_cancer  = st.checkbox("🎗️ Cancer")
            fh_kidney  = st.checkbox("🫘 Kidney Disease")
            fh_obesity = st.checkbox("⚖️ Obesity")
            fh_mental  = st.checkbox("🧘 Mental Health Conditions")

        family_history = []
        if fh_diabetes:     family_history.append("Diabetes")
        if fh_heart:        family_history.append("Heart Disease")
        if fh_stroke:       family_history.append("Stroke")
        if fh_hypertension: family_history.append("Hypertension")
        if fh_cancer:       family_history.append("Cancer")
        if fh_kidney:       family_history.append("Kidney Disease")
        if fh_obesity:      family_history.append("Obesity")
        if fh_mental:       family_history.append("Mental Health")

        if family_history:
            st.info(f"Family history noted: **{', '.join(family_history)}**", icon="👨‍👩‍👧")

        st.markdown("### Existing Medical Conditions")
        st.caption("Conditions you have already been **diagnosed** with.")
        col1, col2 = st.columns(2)
        with col1:
            ec_diabetes  = st.checkbox("Type 2 Diabetes")
            ec_htn       = st.checkbox("High Blood Pressure")
            ec_heart     = st.checkbox("Cardiovascular Disease")
        with col2:
            ec_thyroid   = st.checkbox("Thyroid Disorder")
            ec_arthritis = st.checkbox("Arthritis")
            ec_asthma    = st.checkbox("Asthma / COPD")

        existing_conditions = []
        if ec_diabetes:  existing_conditions.append("Type 2 Diabetes")
        if ec_htn:       existing_conditions.append("Hypertension")
        if ec_heart:     existing_conditions.append("Cardiovascular Disease")
        if ec_thyroid:   existing_conditions.append("Thyroid Disorder")
        if ec_arthritis: existing_conditions.append("Arthritis")
        if ec_asthma:    existing_conditions.append("Asthma / COPD")

        st.markdown("### Current Medications")
        meds_text = st.text_area(
            "List any regular medications (optional)",
            placeholder="e.g. Metformin 500mg, Lisinopril 10mg, Aspirin 75mg...",
            height=80,
        )
        medications = [m.strip() for m in meds_text.split(",") if m.strip()] if meds_text else []

    # ── Tab 4 : Lab Results ───────────────────────────────────────────────────
    with tab4:
        st.markdown("### Blood Test Results *(optional — leave at 0 if unknown)*")
        st.caption(
            "If you have recent lab results, entering them significantly improves accuracy."
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            fasting_glucose = st.number_input(
                "Fasting Glucose (mg/dL)", min_value=0.0, max_value=600.0, value=0.0, step=1.0,
                help="Normal: 70-99 mg/dL. Enter 0 if not measured.",
            )
        with col2:
            total_cholesterol = st.number_input(
                "Total Cholesterol (mg/dL)", min_value=0.0, max_value=600.0, value=0.0, step=1.0,
                help="Desirable: < 200 mg/dL. Enter 0 if not measured.",
            )
        with col3:
            hdl_cholesterol = st.number_input(
                "HDL Cholesterol (mg/dL)", min_value=0.0, max_value=150.0, value=0.0, step=1.0,
                help="Good: > 60 mg/dL. Enter 0 if not measured.",
            )

        if fasting_glucose > 0:
            if fasting_glucose < 100:
                st.success("🟢 Fasting glucose: **Normal** (< 100 mg/dL)", icon="✅")
            elif fasting_glucose < 126:
                st.warning("🟡 Fasting glucose: **Pre-diabetic range** (100–125 mg/dL)", icon="⚠️")
            else:
                st.error("🔴 Fasting glucose: **Diabetic range** (≥ 126 mg/dL) — consult your doctor.", icon="🚨")

        if total_cholesterol > 0:
            if total_cholesterol < 200:
                st.success("🟢 Total cholesterol: **Desirable** (< 200 mg/dL)", icon="✅")
            elif total_cholesterol < 240:
                st.warning("🟡 Total cholesterol: **Borderline high** (200–239 mg/dL)", icon="⚠️")
            else:
                st.error("🔴 Total cholesterol: **High** (≥ 240 mg/dL) — consult your doctor.", icon="🚨")

        if hdl_cholesterol > 0:
            if hdl_cholesterol >= 60:
                st.success(f"🟢 HDL cholesterol: **Optimal** ({hdl_cholesterol} mg/dL)", icon="✅")
            elif hdl_cholesterol >= 40:
                st.info(f"🟡 HDL cholesterol: **Acceptable** ({hdl_cholesterol} mg/dL)")
            else:
                st.warning(f"🟠 HDL cholesterol: **Low** ({hdl_cholesterol} mg/dL) — increased cardiovascular risk.", icon="⚠️")

    # ── Analyse Button ────────────────────────────────────────────────────────
    st.divider()
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        analyse_btn = st.button(
            "🔍 Analyse My Health Risk",
            type="primary",
            use_container_width=True,
        )

    if analyse_btn:
        if not _api_key():
            st.error(
                "⚠️ Please enter your Gemini API key in the sidebar to proceed.",
                icon="🔑",
            )
        else:
            # Build profile
            profile = HealthProfile(
                age=age,
                sex=sex,
                ethnicity=ethnicity,
                bmi=round(bmi_auto, 2),
                systolic_bp=systolic_bp,
                diastolic_bp=diastolic_bp,
                fasting_glucose=fasting_glucose,
                total_cholesterol=total_cholesterol,
                hdl_cholesterol=hdl_cholesterol,
                resting_heart_rate=resting_hr,
                smoking_status=smoking_status,
                alcohol_units_per_week=alcohol_units,
                physical_activity=physical_activity,
                diet_quality=diet_quality,
                sleep_hours=sleep_hours,
                stress_level=stress_level,
                family_history=family_history,
                existing_conditions=existing_conditions,
                medications=medications,
            )

            # Compute rule-based scores instantly
            with st.spinner("⚙️ Computing risk scores..."):
                result = compute_risk(profile)

            # Generate AI report
            with st.spinner("🤖 Gemini 3.6 Flash is analysing your health profile..."):
                try:
                    ai_report = generate_health_report(profile, result, _api_key())
                    st.session_state.ai_report = ai_report
                    st.session_state.health_profile = profile
                    st.session_state.risk_result = result
                    st.session_state.assessment_done = True
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ AI report generation failed: {e}", icon="🚨")
                    st.info(
                        "Tip: Verify your API key and ensure **Gemini 3.6 Flash** is "
                        "accessible in your region. Get a free key at "
                        "https://aistudio.google.com",
                        icon="💡",
                    )


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS SECTION
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.assessment_done:
    result  = st.session_state.risk_result
    profile = st.session_state.health_profile
    report  = st.session_state.ai_report

    # ── Hero risk banner ──────────────────────────────────────────────────────
    risk_level = result.risk_level
    banner_icons = {
        "Low":       "✅",
        "Moderate":  "⚠️",
        "High":      "🚨",
        "Very High": "🔴",
    }
    icon = banner_icons.get(risk_level, "ℹ️")

    banner_fn = {
        "Low":       st.success,
        "Moderate":  st.warning,
        "High":      st.error,
        "Very High": st.error,
    }.get(risk_level, st.info)

    banner_fn(
        f"{icon} **Overall Risk Level: {risk_level} — Score {result.composite_score}/100**  \n"
        f"Assessment for {profile.sex}, Age {profile.age} · "
        f"BMI {profile.bmi:.1f} · BP {profile.systolic_bp}/{profile.diastolic_bp} mmHg",
        icon=icon,
    )

    # ── KPI metrics row ───────────────────────────────────────────────────────
    top_disease = max(result.disease_scores, key=result.disease_scores.get)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🎯 Composite Risk Score", f"{result.composite_score}/100", risk_level)
    m2.metric(
        "⚠️ Highest Risk Area",
        top_disease,
        f"{result.disease_scores[top_disease]:.0f}/100",
    )
    m3.metric("🚨 Risk Factors Identified", len(result.top_risk_factors))
    m4.metric("✅ Protective Factors", len(result.protective_factors))

    st.divider()

    # ── Charts row 1: Gauge + Radar ───────────────────────────────────────────
    st.markdown("### 📊 Risk Visualisation")
    col_g, col_r = st.columns([1, 1])
    with col_g:
        st.plotly_chart(gauge_chart(result.composite_score, result.risk_level), use_container_width=True)
    with col_r:
        st.plotly_chart(radar_chart(result.disease_scores), use_container_width=True)

    # ── Charts row 2: Bar + Donut ─────────────────────────────────────────────
    col_b, col_d = st.columns([3, 2])
    with col_b:
        st.plotly_chart(disease_bar_chart(result.disease_scores), use_container_width=True)
    with col_d:
        st.plotly_chart(risk_factor_donut(result.top_risk_factors, result.protective_factors), use_container_width=True)

    # ── Charts row 3: BMI + BP ────────────────────────────────────────────────
    col_bmi, col_bp = st.columns(2)
    with col_bmi:
        st.plotly_chart(bmi_gauge(profile.bmi), use_container_width=True)
    with col_bp:
        st.plotly_chart(bp_chart(profile.systolic_bp, profile.diastolic_bp), use_container_width=True)

    st.divider()

    # ── Risk & Protective Factors ─────────────────────────────────────────────
    st.markdown("### 🔍 Identified Health Factors")
    col_rf, col_pf = st.columns(2)
    with col_rf:
        st.markdown("**⚠️ Risk Factors**")
        if result.top_risk_factors:
            for f in result.top_risk_factors:
                st.error(f"• {f}", icon="⚠️")
        else:
            st.success("No major risk factors identified!", icon="✅")
    with col_pf:
        st.markdown("**✅ Protective Factors**")
        if result.protective_factors:
            for f in result.protective_factors:
                st.success(f"• {f}", icon="✅")
        else:
            st.info("Focus on building protective habits.", icon="💪")

    st.divider()

    # ── AI-Generated Report ───────────────────────────────────────────────────
    st.markdown("### 🤖 AI Health Assessment & Recommendations")
    st.caption("Generated by **Gemini 3.6 Flash** based on your personal health profile.")
    with st.container(border=True):
        st.markdown(report)

    st.divider()

    # ── Disease Score Summary Table ───────────────────────────────────────────
    st.markdown("### 📋 Disease Risk Score Summary")
    import pandas as pd

    def _risk_badge(s: float) -> str:
        if s < 25:  return "🟢 Low"
        if s < 50:  return "🟡 Moderate"
        if s < 72:  return "🟠 High"
        return "🔴 Very High"

    table_data = pd.DataFrame([
        {
            "Condition":   disease,
            "Risk Score":  f"{score:.0f}/100",
            "Risk Level":  _risk_badge(score),
            "Priority":    "⬆️ High" if score >= 50 else ("➡️ Medium" if score >= 25 else "⬇️ Low"),
        }
        for disease, score in sorted(result.disease_scores.items(), key=lambda x: -x[1])
    ])
    st.dataframe(table_data, use_container_width=True, hide_index=True)

    st.divider()

    # ── AI Health Chat ─────────────────────────────────────────────────────────
    st.markdown("### 💬 Ask Your AI Health Advisor")
    st.caption(
        "Have follow-up questions? Ask **Gemini 3.6 Flash** anything about your health "
        "profile, risk factors, nutrition, or prevention strategies."
    )

    # Display existing chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Quick suggestion chips (only shown before any chat)
    if not st.session_state.chat_history:
        st.markdown("**💡 Suggested questions:**")
        top_risk_name = max(result.disease_scores, key=result.disease_scores.get)
        suggestions = [
            f"What is the single most important change I can make to reduce my {top_risk_name} risk?",
            "Can you create a 7-day meal plan tailored to my health profile?",
            f"What exercises are best for a {profile.age}-year-old with {profile.physical_activity} activity level?",
        ]
        sc1, sc2, sc3 = st.columns(3)
        for col, suggestion in zip([sc1, sc2, sc3], suggestions):
            with col:
                if st.button(
                    suggestion[:65] + "…",
                    use_container_width=True,
                    key=f"chip_{suggestion[:25]}",
                ):
                    st.session_state.chat_history.append({"role": "user", "content": suggestion})
                    with st.spinner("Gemini is thinking..."):
                        try:
                            answer = ask_health_question(
                                suggestion, profile, result,
                                st.session_state.chat_history, _api_key(),
                            )
                            st.session_state.chat_history.append(
                                {"role": "assistant", "content": answer}
                            )
                        except Exception as e:
                            st.session_state.chat_history.append(
                                {"role": "assistant", "content": f"Sorry, I encountered an error: {e}"}
                            )
                    st.rerun()

    # Chat input
    if user_q := st.chat_input("Ask about your health risks, prevention, diet, exercise..."):
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)
        with st.chat_message("assistant"):
            with st.spinner("Gemini 3.6 Flash is thinking..."):
                try:
                    answer = ask_health_question(
                        user_q, profile, result,
                        st.session_state.chat_history, _api_key(),
                    )
                    st.markdown(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    err_msg = f"Sorry, I couldn't process that: {e}"
                    st.error(err_msg, icon="🚨")
                    st.session_state.chat_history.append({"role": "assistant", "content": err_msg})

    # Clear chat
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

    st.divider()
    st.caption(
        "🩺 **Chronic Disease Risk Predictor** · Powered by Gemini 3.6 Flash · "
        "For educational purposes only — not a substitute for professional medical advice."
    )
