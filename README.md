# 🩺 Chronic Disease Risk Prediction Dashboard

> **AI-Powered** health risk assessment using **Gemini 3.6 Flash** and **Streamlit**  
> No HTML · No CSS · No JavaScript — 100% Python

---

## Overview

A fully deployable Streamlit application that lets users input their lifestyle and basic health data and receive:

- **Composite risk score (0–100)** across 6 chronic diseases
- **Disease-specific breakdowns** with interactive Plotly charts
- **AI-generated personalised health report** from Gemini 3.6 Flash
- **Prioritised action plan** covering the next 30 days, 1–3 months, and 3–12 months
- **Nutrition guidance & exercise prescription** tailored to the user's profile
- **Interactive AI health chat** for unlimited follow-up questions

---

## Project Structure

```
chronic-disease-risk-prediction/
├── app.py              # Main Streamlit application (UI + routing)
├── risk_engine.py      # Rule-based risk scoring engine (0-100 scores)
├── gemini_advisor.py   # Gemini 3.6 Flash AI integration (new google-genai SDK)
├── charts.py           # Plotly chart builders (6 chart types)
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
└── README.md           # This file
```

---

## Quick Start

### 1. Clone / download this project

```bash
git clone <repo-url>
cd chronic-disease-risk-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Requires Python 3.10+**

### 3. Get a Gemini API Key

Get a **free** key at [Google AI Studio](https://aistudio.google.com).

### 4. Configure your API key

**Option A — `.env` file (recommended for local development):**
```bash
cp .env.example .env
# Then edit .env and set your key:
# GEMINI_API_KEY=AIza...
```

**Option B — Enter directly in the app sidebar** *(no file needed — key is never stored)*

### 5. Run the app

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## How It Works

### Rule-Based Risk Engine (`risk_engine.py`)

Computes a **0–100 composite risk score** using clinical heuristics across 6 chronic diseases.
An **age multiplier** scales scores upward with age.

| Disease | Key Scoring Factors |
|---|---|
| Type 2 Diabetes | BMI, fasting glucose, activity, diet, stress, ethnicity, family history |
| Cardiovascular Disease | BP, cholesterol, smoking, BMI, resting HR, family history |
| Hypertension | BP, BMI, stress, alcohol, activity, family history |
| Obesity / Metabolic | BMI, activity, diet, sleep, stress |
| Mental Health | Stress, sleep, activity, alcohol, smoking |
| Chronic Kidney Disease | BP, glucose, BMI, existing diabetes/hypertension |

Risk levels: **Low** (0–24) · **Moderate** (25–49) · **High** (50–71) · **Very High** (72–100)

### Gemini 3.6 Flash AI Report (`gemini_advisor.py`)

Uses the new **`google-genai`** SDK. The structured prompt sends the full health profile + computed scores to Gemini, which returns a **markdown report** with:

1. 🩺 Health Risk Summary
2. ⚠️ Top Health Risks & Why
3. ✅ Preventive Action Plan (30-day / 1–3 month / 3–12 month)
4. 🥗 Personalised Nutrition Guidance
5. 🏃 Exercise Prescription
6. 🧠 Mental & Emotional Wellbeing
7. 🔬 Recommended Health Screenings
8. 💡 Motivational Summary

### Interactive AI Chat

After the report, users can ask unlimited follow-up questions to a context-aware Gemini 3.6 Flash assistant that remembers their full health profile throughout the conversation (true multi-turn via `google-genai` Contents API).

### Charts (`charts.py`)

All built with **Plotly** — pure Python, zero HTML/CSS/JS:

| Chart | Purpose |
|---|---|
| Gauge (composite) | Overall risk score with colour-coded zones |
| Radar | Multi-disease risk profile spider chart |
| Horizontal bar | Disease-specific scores with threshold markers |
| Donut | Risk vs protective factor ratio |
| BMI gauge | BMI category with WHO zones |
| BP bar chart | Systolic/diastolic vs normal and hypertension thresholds |

---

## Input Data

| Tab | Fields |
|---|---|
| Demographics & Biometrics | Age, sex, ethnicity, weight, height (auto-BMI), blood pressure, heart rate |
| Lifestyle & Habits | Physical activity, diet quality, smoking, alcohol, sleep, stress level |
| Medical History | Family history (8 conditions), existing diagnoses (6), current medications |
| Lab Results (optional) | Fasting glucose, total cholesterol, HDL cholesterol |

---

## Deployment

### Streamlit Community Cloud (free, recommended)

1. Push this repo to **GitHub** (all files)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo, set `app.py` as the entry point
4. Add your API key under **Settings → Secrets**:
   ```toml
   GEMINI_API_KEY = "AIza..."
   ```
5. Deploy — the app automatically reads `st.secrets["GEMINI_API_KEY"]`

### Other platforms (Heroku, Railway, Render, etc.)

Set `GEMINI_API_KEY` as an environment variable in the platform dashboard.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | ≥ 1.35 | Web application framework |
| `google-genai` | ≥ 1.0 | Gemini 3.6 Flash API (new SDK) |
| `plotly` | ≥ 5.22 | Interactive charts |
| `pandas` | ≥ 2.2 | Data tables |
| `numpy` | ≥ 1.26 | Numerical operations |
| `python-dotenv` | ≥ 1.0 | `.env` file loading |

> **Note:** This project uses the **new `google-genai` SDK** (package: `google-genai`), not the deprecated `google-generativeai` package. Do not mix both in the same environment.

---

## Disclaimer

This tool is for **educational and informational purposes only**. It does **not** constitute medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for medical decisions.
