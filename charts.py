"""
charts.py
Plotly-based chart builders for the risk dashboard.
All charts are pure Python / Plotly — no HTML/CSS/JS.
"""

import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

RISK_COLORS = {
    "Low":       "#22c55e",
    "Moderate":  "#f59e0b",
    "High":      "#f97316",
    "Very High": "#ef4444",
}

# Colour each disease bar based on score
def _bar_color(score: float) -> str:
    if score < 25:  return RISK_COLORS["Low"]
    if score < 50:  return RISK_COLORS["Moderate"]
    if score < 72:  return RISK_COLORS["High"]
    return RISK_COLORS["Very High"]


_LAYOUT_DEFAULTS = dict(
    paper_bgcolor="white",
    font={"family": "system-ui, -apple-system, sans-serif", "size": 12},
)


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def gauge_chart(score: float, risk_level: str) -> go.Figure:
    """Composite risk score gauge with colour-coded zones."""
    color = RISK_COLORS.get(risk_level, "#6b7280")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        title={"text": "Overall Health Risk Score", "font": {"size": 18}},
        delta={
            "reference": 50,
            "increasing": {"color": "#ef4444"},
            "decreasing": {"color": "#22c55e"},
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "#9ca3af",
                "tickvals": [0, 25, 50, 72, 100],
                "ticktext": ["0", "25\nLow", "50\nMod", "72\nHigh", "100"],
            },
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "#e5e7eb",
            "steps": [
                {"range": [0, 25],   "color": "#dcfce7"},
                {"range": [25, 50],  "color": "#fef9c3"},
                {"range": [50, 72],  "color": "#ffedd5"},
                {"range": [72, 100], "color": "#fee2e2"},
            ],
            "threshold": {
                "line": {"color": "#1f2937", "width": 4},
                "thickness": 0.75,
                "value": score,
            },
        },
        number={"font": {"size": 38, "color": color}, "suffix": "/100"},
    ))
    fig.update_layout(
        height=300,
        margin=dict(t=60, b=20, l=40, r=40),
        **_LAYOUT_DEFAULTS,
    )
    return fig


def disease_bar_chart(disease_scores: Dict[str, float]) -> go.Figure:
    """Horizontal bar chart of disease-specific risk scores."""
    df = pd.DataFrame(
        list(disease_scores.items()), columns=["Disease", "Score"]
    ).sort_values("Score", ascending=True)

    colors = [_bar_color(s) for s in df["Score"]]

    fig = go.Figure(go.Bar(
        x=df["Score"],
        y=df["Disease"],
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{s:.0f}" for s in df["Score"]],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Risk Score: %{x:.1f}/100<extra></extra>",
    ))

    # Reference lines
    for xval, label, lcolor in [(25, "Low/Mod", "#86efac"), (50, "Mod/High", "#fcd34d"), (72, "High/VHigh", "#fb923c")]:
        fig.add_vline(
            x=xval, line_dash="dot", line_color=lcolor, line_width=1.5,
            annotation_text=label, annotation_position="top",
            annotation_font_size=9, annotation_font_color="#6b7280",
        )

    fig.update_layout(
        title={"text": "Disease-Specific Risk Breakdown", "font": {"size": 15}},
        xaxis=dict(range=[0, 115], title="Risk Score (0–100)", showgrid=True, gridcolor="#f3f4f6"),
        yaxis=dict(title=""),
        height=330,
        margin=dict(t=50, b=40, l=20, r=70),
        plot_bgcolor="white",
        **_LAYOUT_DEFAULTS,
    )
    return fig


def radar_chart(disease_scores: Dict[str, float]) -> go.Figure:
    """Radar/spider chart for multi-disease risk profile."""
    categories = list(disease_scores.keys())
    values = list(disease_scores.values())
    # Close the polygon
    cats_closed = categories + [categories[0]]
    vals_closed = values + [values[0]]

    fig = go.Figure()
    # Reference shape at score=50
    ref_vals = [50] * (len(categories) + 1)
    fig.add_trace(go.Scatterpolar(
        r=ref_vals,
        theta=cats_closed,
        fill="toself",
        fillcolor="rgba(239,68,68,0.05)",
        line=dict(color="#fca5a5", width=1, dash="dot"),
        name="Moderate threshold (50)",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatterpolar(
        r=vals_closed,
        theta=cats_closed,
        fill="toself",
        fillcolor="rgba(59,130,246,0.18)",
        line=dict(color="#3b82f6", width=2.5),
        marker=dict(size=7, color="#3b82f6"),
        name="Your risk profile",
        hovertemplate="<b>%{theta}</b><br>Score: %{r:.1f}<extra></extra>",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont={"size": 9},
                gridcolor="#e5e7eb",
                tickvals=[25, 50, 75, 100],
            ),
            angularaxis=dict(tickfont={"size": 10}),
            bgcolor="white",
        ),
        title={"text": "Risk Profile Radar", "font": {"size": 15}},
        showlegend=True,
        legend=dict(orientation="h", y=-0.15, font={"size": 9}),
        height=400,
        margin=dict(t=60, b=50, l=40, r=40),
        **_LAYOUT_DEFAULTS,
    )
    return fig


def risk_factor_donut(risk_factors: List[str], protective_factors: List[str]) -> go.Figure:
    """Donut chart comparing risk factor count vs protective factor count."""
    n_risk = max(len(risk_factors), 0)
    n_prot = max(len(protective_factors), 0)

    # Avoid zero-slice
    vals = [max(n_risk, 1), max(n_prot, 1)]
    colors = ["#f97316", "#22c55e"]

    fig = go.Figure(go.Pie(
        labels=["Risk Factors", "Protective Factors"],
        values=vals,
        hole=0.58,
        marker=dict(colors=colors, line=dict(color="white", width=2.5)),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b>: %{value} factors<extra></extra>",
        pull=[0.04, 0],
    ))
    fig.add_annotation(
        text=f"{n_risk}R / {n_prot}P",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color="#374151"),
    )
    fig.update_layout(
        title={"text": "Risk vs Protective Factors", "font": {"size": 14}},
        height=295,
        margin=dict(t=50, b=30, l=20, r=20),
        showlegend=False,
        **_LAYOUT_DEFAULTS,
    )
    return fig


def bmi_gauge(bmi: float) -> go.Figure:
    """BMI gauge with WHO category colour coding."""
    if bmi < 18.5:
        category, color = "Underweight", "#60a5fa"
    elif bmi < 25:
        category, color = "Normal", "#22c55e"
    elif bmi < 30:
        category, color = "Overweight", "#f59e0b"
    elif bmi < 35:
        category, color = "Obese Class I", "#f97316"
    else:
        category, color = "Obese Class II+", "#ef4444"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=bmi,
        title={"text": f"BMI — {category}", "font": {"size": 14}},
        gauge={
            "axis": {"range": [10, 45], "tickwidth": 1, "tickvals": [10, 18.5, 25, 30, 35, 45]},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "white",
            "steps": [
                {"range": [10, 18.5], "color": "#dbeafe"},
                {"range": [18.5, 25], "color": "#dcfce7"},
                {"range": [25, 30],   "color": "#fef3c7"},
                {"range": [30, 35],   "color": "#ffedd5"},
                {"range": [35, 45],   "color": "#fee2e2"},
            ],
        },
        number={"suffix": " kg/m²", "font": {"size": 22, "color": color}},
    ))
    fig.update_layout(
        height=230,
        margin=dict(t=50, b=10, l=30, r=30),
        **_LAYOUT_DEFAULTS,
    )
    return fig


def bp_chart(systolic: int, diastolic: int) -> go.Figure:
    """Blood pressure bar chart with normal and hypertension threshold lines."""
    categories = ["Systolic (mmHg)", "Diastolic (mmHg)"]
    values = [systolic, diastolic]

    sys_color = "#ef4444" if systolic >= 140 else ("#f59e0b" if systolic >= 130 else "#22c55e")
    dia_color = "#ef4444" if diastolic >= 90 else ("#f59e0b" if diastolic >= 85 else "#22c55e")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Your Reading",
        x=categories,
        y=values,
        marker_color=[sys_color, dia_color],
        text=[f"<b>{v} mmHg</b>" for v in values],
        textposition="outside",
        width=0.45,
        hovertemplate="<b>%{x}</b>: %{y} mmHg<extra></extra>",
    ))
    # Normal threshold line
    fig.add_trace(go.Scatter(
        name="Normal limit (120/80)",
        x=categories,
        y=[120, 80],
        mode="markers+lines",
        line=dict(color="#22c55e", dash="dash", width=1.5),
        marker=dict(size=9, symbol="diamond"),
    ))
    # Hypertension threshold line
    fig.add_trace(go.Scatter(
        name="Hypertension (140/90)",
        x=categories,
        y=[140, 90],
        mode="markers+lines",
        line=dict(color="#ef4444", dash="dash", width=1.5),
        marker=dict(size=9, symbol="diamond"),
    ))

    fig.update_layout(
        title={"text": "Blood Pressure Reading", "font": {"size": 14}},
        yaxis=dict(range=[0, max(200, systolic + 40)], title="mmHg", gridcolor="#f3f4f6"),
        height=270,
        margin=dict(t=50, b=40, l=50, r=20),
        plot_bgcolor="white",
        legend=dict(orientation="h", y=-0.28, font={"size": 10}),
        bargap=0.35,
        **_LAYOUT_DEFAULTS,
    )
    return fig
