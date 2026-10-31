"""
Shared UI Components, Visual Cards, and Plotly Chart Builders for Streamlit Hub.
"""

from typing import Dict, List
import numpy as np
import plotly.graph_objects as go
import streamlit as st


def render_metric_card(label: str, value: str, subtext: str = "", border_color: str = "#3b82f6"):
    """Render a styled metric card."""
    st.markdown(
        f"""
        <div style="background-color: #f8fafc; border-left: 4px solid {border_color};
                    border-radius: 8px; padding: 14px 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 12px;">
            <div style="font-size: 0.8rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">
                {label}
            </div>
            <div style="font-size: 1.6rem; font-weight: 700; color: #0f172a; margin-top: 4px;">
                {value}
            </div>
            {f'<div style="font-size: 0.8rem; color: #94a3b8; margin-top: 2px;">{subtext}</div>' if subtext else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_gauge_chart(probability: float, title: str = "Churn Risk Probability") -> go.Figure:
    """Build a gauge chart showing risk tier zones."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 28, "color": "#0f172a", "family": "Inter"}},
            title={"text": f"<b>{title}</b>", "font": {"size": 14, "color": "#1e293b"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                "bar": {"color": "#1e293b", "thickness": 0.25},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#cbd5e1",
                "steps": [
                    {"range": [0, 35], "color": "#dcfce7"},     # Low (Green)
                    {"range": [35, 70], "color": "#fef9c3"},    # Moderate (Yellow)
                    {"range": [70, 100], "color": "#fee2e2"},   # High (Red)
                ],
                "threshold": {
                    "line": {"color": "#dc2626", "width": 4},
                    "thickness": 0.8,
                    "value": 70,
                },
            },
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
    )
    return fig


def build_shap_bar_chart(shap_values: Dict[str, float]) -> go.Figure:
    """Horizontal bar chart for local SHAP feature attributions."""
    sorted_items = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
    sorted_items.reverse()  # Ascending for bottom-to-top rendering

    features = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    colors = ["#ef4444" if v >= 0 else "#10b981" for v in values]
    text_labels = [f"{v:+.3f}" for v in values]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=features,
            orientation="h",
            text=text_labels,
            textposition="outside",
            marker=dict(color=colors, line=dict(color="#334155", width=1)),
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#64748b", line_width=1.2)
    fig.update_layout(
        title=dict(text="<b>Local SHAP Feature Attributions</b>", font=dict(size=13, color="#1e293b")),
        xaxis=dict(title="Contribution to Churn Log-Odds", showgrid=True, gridcolor="#e2e8f0"),
        yaxis=dict(autorange=True),
        margin=dict(l=120, r=30, t=40, b=40),
        height=260,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    return fig


def build_probability_bar_chart(predictions: List[Dict], title: str = "Probability Breakdown") -> go.Figure:
    """Horizontal bar chart for multi-class prediction probabilities."""
    sorted_preds = sorted(predictions, key=lambda x: x["probability"])
    labels = [p["label"] for p in sorted_preds]
    probs = [p["probability"] for p in sorted_preds]
    texts = [f"{p * 100:.1f}%" for p in probs]

    colors = ["#93c5fd"] * len(probs)
    if colors:
        colors[-1] = "#2563eb"  # Highest probability in dark cobalt

    fig = go.Figure(
        go.Bar(
            x=probs,
            y=labels,
            orientation="h",
            text=texts,
            textposition="outside",
            marker=dict(color=colors, line=dict(color="#1e293b", width=1)),
        )
    )
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=13, color="#1e293b")),
        xaxis=dict(title="Confidence", range=[0, 1.15], tickformat=".0%", showgrid=True, gridcolor="#e2e8f0"),
        margin=dict(l=90, r=40, t=40, b=40),
        height=240,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    return fig


def build_cartpole_trajectory_chart(state_history: List[List[float]], q_history: List[List[float]]) -> go.Figure:
    """Multi-trace chart showing Cart Position, Pole Angle, and Q-values across timesteps."""
    steps = list(range(len(state_history)))
    cart_pos = [s[0] for s in state_history]
    pole_angle = [np.degrees(s[2]) for s in state_history]  # Convert radians to degrees
    q_left = [q[0] for q in q_history]
    q_right = [q[1] for q in q_history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=pole_angle, mode="lines", name="Pole Angle (deg)", line=dict(color="#ef4444", width=2)))
    fig.add_trace(go.Scatter(x=steps, y=cart_pos, mode="lines", name="Cart Position (m)", line=dict(color="#3b82f6", width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=steps, y=q_left, mode="lines", name="Q(Left)", line=dict(color="#10b981", width=1.5)))
    fig.add_trace(go.Scatter(x=steps, y=q_right, mode="lines", name="Q(Right)", line=dict(color="#8b5cf6", width=1.5)))

    fig.add_hline(y=12.0, line_dash="dash", line_color="#dc2626", line_width=1, annotation_text="+12° Limit")
    fig.add_hline(y=-12.0, line_dash="dash", line_color="#dc2626", line_width=1, annotation_text="-12° Limit")

    fig.update_layout(
        title=dict(text="<b>CartPole-v1 State Dynamics & Policy Q-Values</b>", font=dict(size=14, color="#1e293b")),
        xaxis=dict(title="Simulation Step", showgrid=True, gridcolor="#e2e8f0"),
        yaxis=dict(title="Magnitude / Q-Value", showgrid=True, gridcolor="#e2e8f0"),
        margin=dict(l=60, r=40, t=50, b=40),
        height=320,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig

