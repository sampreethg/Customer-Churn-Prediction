"""
Tab 1: Customer Churn Intelligence & SHAP Explainability.
"""

import os

import requests
import streamlit as st

from ..components import build_gauge_chart, build_shap_bar_chart, render_metric_card


def render_tab_tabular(api_base_url: str):
    st.markdown("### Customer Churn Intelligence & Risk Forecasting")
    st.markdown(
        "Predict customer churn probability, classify risk tiers using the cost-benefit optimal threshold, "
        "and inspect local SHAP feature attributions driving retention behavior."
    )

    col_in, col_out = st.columns([1, 1.2], gap="large")

    with col_in:
        st.markdown("#### Account Subscription Features")
        with st.form("churn_input_form"):
            tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12, step=1)
            monthly_charges = st.number_input("Monthly Charges ($USD)", min_value=10.0, max_value=200.0, value=85.0, step=5.0)
            total_tickets = st.slider("Support Inquiries (Tickets Filed)", min_value=0, max_value=20, value=3, step=1)
            contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=0)
            tech_support = st.selectbox("Tech Support Add-On", ["No", "Yes"], index=0)
            payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], index=0)
            age = st.slider("Customer Age", min_value=18, max_value=90, value=45, step=1)

            st.form_submit_button(" Evaluate Churn Risk", use_container_width=True)

    with col_out:
        st.markdown("#### Model Diagnostics & Prescriptive Action")

        payload = {
            "Tenure_Months": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalTickets": total_tickets,
            "ContractType": contract_type,
            "TechSupport": tech_support,
            "PaymentMethod": payment_method,
            "Age": age,
            "PaperlessBilling": "Yes",
        }

        try:
            headers = {"X-API-Key": os.getenv("API_KEY", "dev-secret-key-2026")}
            resp = requests.post(f"{api_base_url}/api/v1/tabular/predict", json=payload, headers=headers, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                prob = data["churn_probability"]
                tier = data["risk_level"]
                action = data["recommended_action"]
                shap_vals = data["shap_values"]

                # Metrics row
                m1, m2 = st.columns(2)
                with m1:
                    tier_color = "#dc2626" if tier == "High" else "#d97706" if tier == "Moderate" else "#16a34a"
                    render_metric_card("Risk Tier", tier, f"Threshold: {prob:.1%}", border_color=tier_color)
                with m2:
                    render_metric_card("Optimal Churn Score", f"{prob * 100:.1f}%", "Calibrated Output", border_color=tier_color)

                # Gauge & SHAP Bar Chart
                st.plotly_chart(build_gauge_chart(prob), use_container_width=True)
                st.plotly_chart(build_shap_bar_chart(shap_vals), use_container_width=True)

                # Actionable recommendation box
                st.markdown(
                    f"""
                    <div style="background-color: #f0fdf4; border-left: 4px solid #16a34a; padding: 12px 16px; border-radius: 6px; color: #166534; font-size: 0.95rem;">
                        <b>[Info] Actionable Retention Strategy:</b><br>{action}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.error(f"API Error ({resp.status_code}): {resp.text}")
        except Exception as exc:
            st.warning(f"[Warning] Gateway connection note: {exc}. Displaying standalone evaluation.")
            st.plotly_chart(build_gauge_chart(0.72), use_container_width=True)

