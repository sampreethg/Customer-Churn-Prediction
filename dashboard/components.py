"""
Streamlit UI Components, Interactive Visualizations, Risk Card Renderers,
and SHAP Explainability Plots.
"""

import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def render_risk_gauge(probability: float):
    """
    Build an interactive Plotly indicator/gauge chart (0% to 100%) with custom color steps,
    or render an SVG gauge fallback if Plotly is not installed.

    Parameters
    ----------
    probability : float
        Churn probability (0.0 to 1.0).
    """
    pct_value = round(probability * 100.0, 1)

    if pct_value >= 70.0:
        bar_color = "#dc2626"
        tier_label = "High Risk"
    elif pct_value >= 35.0:
        bar_color = "#d97706"
        tier_label = "Medium Risk"
    else:
        bar_color = "#16a34a"
        tier_label = "Low Risk"

    if HAS_PLOTLY:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=pct_value,
                number={
                    "suffix": "%",
                    "font": {"size": 38, "color": bar_color, "family": "Inter, sans-serif"},
                },
                title={
                    "text": "<b>Predicted Churn Risk Probability</b>",
                    "font": {"size": 18, "color": "#1e293b", "family": "Inter, sans-serif"},
                },
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                    "bar": {"color": bar_color, "thickness": 0.28},
                    "bgcolor": "white",
                    "borderwidth": 1,
                    "bordercolor": "#cbd5e1",
                    "steps": [
                        {"range": [0, 35], "color": "#dcfce7"},
                        {"range": [35, 70], "color": "#fef3c7"},
                        {"range": [70, 100], "color": "#fee2e2"},
                    ],
                    "threshold": {
                        "line": {"color": "#b91c1c", "width": 3},
                        "thickness": 0.85,
                        "value": 70,
                    },
                },
            )
        )
        fig.update_layout(
            height=280,
            margin=dict(l=25, r=25, t=50, b=25),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"family": "Inter, sans-serif"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        gauge_html = f"""
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 10px; padding: 20px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="font-size: 1.1rem; font-weight: 700; color: #1e293b; margin-bottom: 12px;">Predicted Churn Risk Probability</div>
            <div style="font-size: 2.8rem; font-weight: 800; color: {bar_color};">{pct_value}%</div>
            <div style="font-size: 0.95rem; font-weight: 600; color: #64748b; margin-bottom: 14px;">Operational Status: {tier_label}</div>
            <div style="background-color: #f1f5f9; border-radius: 9999px; height: 16px; width: 100%; overflow: hidden; position: relative; border: 1px solid #cbd5e1;">
                <div style="width: {pct_value}%; height: 100%; background-color: {bar_color}; border-radius: 9999px; transition: width 0.4s ease;"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #94a3b8; margin-top: 6px;">
                <span>0% (Low)</span>
                <span>35% (Medium)</span>
                <span>70% (High)</span>
                <span>100%</span>
            </div>
        </div>
        """
        st.markdown(gauge_html, unsafe_allow_html=True)


def render_risk_card(risk_level: str, probability: float, recommendation: str) -> None:
    """
    Display a styled metric card highlighting the risk tier, probability percentage,
    and concrete retention play.
    """
    tier_configs = {
        "High": {
            "border_color": "#ef4444",
            "badge_bg": "#fee2e2",
            "badge_color": "#991b1b",
            "icon": "[Warning]",
            "title": "Immediate Churn Hazard",
        },
        "Medium": {
            "border_color": "#f59e0b",
            "badge_bg": "#fef3c7",
            "badge_color": "#92400e",
            "icon": "",
            "title": "Moderate Attrition Risk",
        },
        "Low": {
            "border_color": "#10b981",
            "badge_bg": "#d1fae5",
            "badge_color": "#065f46",
            "icon": "[OK]",
            "title": "Healthy Account Engagement",
        },
    }

    cfg = tier_configs.get(risk_level, tier_configs["Low"])
    pct_str = f"{probability * 100.0:.1f}%"

    st.markdown(
        f"""
        <div style="
            border-left: 6px solid {cfg["border_color"]};
            background-color: #ffffff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.06);
            margin-bottom: 20px;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span style="font-size: 1.15rem; font-weight: 700; color: #1e293b;">
                    {cfg["icon"]} {cfg["title"]}
                </span>
                <span style="
                    background-color: {cfg["badge_bg"]};
                    color: {cfg["badge_color"]};
                    font-weight: 700;
                    font-size: 0.9rem;
                    padding: 4px 12px;
                    border-radius: 9999px;
                ">
                    {risk_level.upper()} RISK ({pct_str})
                </span>
            </div>
            <div style="margin-top: 8px;">
                <p style="margin: 0; font-size: 0.85rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">
                    Prescribed Retention Policy
                </p>
                <p style="margin: 6px 0 0 0; font-size: 1.02rem; color: #0f172a; font-weight: 500; line-height: 1.5;">
                    {recommendation}
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_batch_summary(df_results: pd.DataFrame) -> None:
    """
    Generate summary KPI metrics alongside a Plotly donut chart depicting
    the risk distribution across the cohort.
    """
    total_accounts = len(df_results)
    high_count = (df_results["risk_level"] == "High").sum()
    med_count = (df_results["risk_level"] == "Medium").sum()
    low_count = (df_results["risk_level"] == "Low").sum()
    avg_score = df_results["churn_probability"].mean() * 100.0 if total_accounts > 0 else 0.0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric(label="Total Accounts Scored", value=f"{total_accounts:,}")
    with kpi2:
        high_pct = (high_count / total_accounts * 100.0) if total_accounts > 0 else 0.0
        st.metric(
            label="High Risk Flight Hazards",
            value=f"{high_count:,}",
            delta=f"{high_pct:.1f}% of cohort",
            delta_color="inverse",
        )
    with kpi3:
        med_pct = (med_count / total_accounts * 100.0) if total_accounts > 0 else 0.0
        st.metric(
            label="Medium Risk Watchlist",
            value=f"{med_count:,}",
            delta=f"{med_pct:.1f}% of cohort",
            delta_color="off",
        )
    with kpi4:
        st.metric(label="Average Cohort Risk Score", value=f"{avg_score:.1f}%")

    st.markdown("---")

    if HAS_PLOTLY:
        dist_df = pd.DataFrame(
            {
                "Risk Tier": ["High Risk", "Medium Risk", "Low Risk"],
                "Count": [high_count, med_count, low_count],
                "Color": ["#ef4444", "#f59e0b", "#10b981"],
            }
        )
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=dist_df["Risk Tier"],
                    values=dist_df["Count"],
                    hole=0.55,
                    marker=dict(colors=dist_df["Color"]),
                    textinfo="label+percent+value",
                    hoverinfo="label+percent+value",
                    textfont=dict(size=14, family="Inter, sans-serif"),
                )
            ]
        )
        fig.update_layout(
            title={
                "text": "<b>Cohort Churn Risk Distribution</b>",
                "font": {"size": 17, "color": "#1e293b", "family": "Inter, sans-serif"},
            },
            height=320,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("#### Cohort Churn Risk Distribution")
        dist_cols = st.columns(3)
        with dist_cols[0]:
            st.info(
                f"[High] **High Risk:** {high_count} accounts ({high_count / total_accounts * 100:.1f}%)"
                if total_accounts > 0
                else "[High] High Risk: 0"
            )
        with dist_cols[1]:
            st.warning(
                f"[Medium] **Medium Risk:** {med_count} accounts ({med_count / total_accounts * 100:.1f}%)"
                if total_accounts > 0
                else "[Medium] Medium Risk: 0"
            )
        with dist_cols[2]:
            st.success(
                f"[Low] **Low Risk:** {low_count} accounts ({low_count / total_accounts * 100:.1f}%)"
                if total_accounts > 0
                else "[Low] Low Risk: 0"
            )


def render_shap_breakdown(explanation_data: dict, top_n: int = 8) -> None:
    """
    Render interactive SHAP local contribution waterfall/force chart and narrative breakdown.

    Parameters
    ----------
    explanation_data : dict
        Explanation dictionary from /explain endpoint or local ChurnExplainer.
    top_n : int, default=8
        Number of top contributing features to plot.
    """
    base_val = explanation_data.get("base_value", 0.50)
    pred_val = explanation_data.get("churn_probability", 0.50)
    narrative = explanation_data.get("summary_narrative", "")
    all_features = explanation_data.get("features", [])

    if not all_features:
        # Fallback if features not nested: check top_risk_drivers and top_retention_drivers
        all_features = explanation_data.get("top_risk_drivers", []) + explanation_data.get(
            "top_retention_drivers", []
        )

    # Sort and take top N contributors
    plot_feats = sorted(all_features, key=lambda x: abs(x.get("shap_value", 0.0)), reverse=True)[
        :top_n
    ]
    plot_feats = list(reversed(plot_feats)) # Ascending for horizontal bar

    names = [f["feature_name"] for f in plot_feats]
    shap_vals = [f["shap_value"] for f in plot_feats]
    bar_colors = ["#ef4444" if val >= 0 else "#10b981" for val in shap_vals]

    # Narrative Callout Box
    st.markdown(
        f"""
        <div style="
            background-color: #f8fafc;
            border-left: 4px solid #3b82f6;
            padding: 14px 18px;
            border-radius: 6px;
            margin-bottom: 16px;
        ">
            <span style="font-weight: 700; color: #1e3a8a; font-size: 0.95rem;">[Info] Automated XAI Narrative:</span>
            <p style="margin: 4px 0 0 0; color: #334155; font-size: 0.98rem; line-height: 1.45;">
                {narrative}
            </p>
            <div style="margin-top: 8px; font-size: 0.85rem; color: #64748b;">
                Population Baseline (Expected): <b>{base_val:.2%}</b> &nbsp;|&nbsp;
                This Customer's Score: <b>{pred_val:.2%}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if HAS_PLOTLY:
        hover_texts = [
            f"<b>{name}</b><br>SHAP Impact: {val:+.4f}<br>{'Pushes toward Churn (+Risk)' if val >= 0 else 'Protects Account (-Risk)'}"
            for name, val in zip(names, shap_vals, strict=False)
        ]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=names,
                x=shap_vals,
                orientation="h",
                marker=dict(color=bar_colors, line=dict(color="#334155", width=1)),
                hovertext=hover_texts,
                hoverinfo="text",
                text=[f"{val:+.3f}" for val in shap_vals],
                textposition="outside",
                textfont=dict(size=11, family="Inter, sans-serif"),
            )
        )

        fig.add_vline(x=0, line_width=1.5, line_dash="dash", line_color="#475569")

        fig.update_layout(
            title={
                "text": f"<b>Local Feature Attributions (SHAP)</b> &nbsp;<span style='font-size:12px;color:#64748b;'>Base: {base_val:.2f} → Score: {pred_val:.2f}</span>",
                "font": {"size": 15, "color": "#0f172a", "family": "Inter, sans-serif"},
            },
            xaxis_title="SHAP Attribution Value (Impact on Probability)",
            yaxis_title="Transformed Feature",
            height=340,
            margin=dict(l=20, r=40, t=50, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(248,250,252,0.6)",
            font=dict(family="Inter, sans-serif"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Fallback styled breakdown table
        st.markdown("#### Feature Attributions Breakdown")
        for f in reversed(plot_feats):
            val = f["shap_value"]
            fname = f["feature_name"]
            if val >= 0:
                st.markdown(f"[High] **{fname}**: `+{val:.4f}` *(increases churn risk)*")
            else:
                st.markdown(f"[Low] **{fname}**: `{val:.4f}` *(reduces churn risk / loyalty signal)*")
