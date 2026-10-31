"""
SHAP Explainability Engine for Customer Churn Modeling.

Provides local customer-level and global model interpretability using TreeExplainer,
post-ColumnTransformer attribution mapping, prescriptive driver extraction,
and localized contribution visualizations.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

from .config import MODEL_PATH


class ChurnExplainer:
    """
    Explainability engine wrapping a fitted Scikit-Learn churn pipeline with SHAP TreeExplainer.
    """

    def __init__(self, model_or_path: Any = None):
        """
        Initialize explainer with pipeline instance or path to serialized joblib model.
        """
        if model_or_path is None:
            model_or_path = MODEL_PATH

        if isinstance(model_or_path, (str, Path)):
            path = Path(model_or_path)
            if not path.is_file():
                raise FileNotFoundError(
                    f"Model artifact not found for explanation: {path.resolve()}"
                )
            self.model = joblib.load(path)
        else:
            self.model = model_or_path

        self.preprocessor = self.model.named_steps["preprocessor"]
        self.classifier = self.model.named_steps["classifier"]
        self.explainer = shap.TreeExplainer(self.classifier)

        # Extract transformed column names
        if hasattr(self.preprocessor, "get_feature_names_out"):
            raw_names = self.preprocessor.get_feature_names_out()
            self.feature_names = [
                name.replace("num__", "").replace("cat__", "") for name in raw_names
            ]
        else:
            self.feature_names = [f"feature_{i}" for i in range(self.classifier.n_features_in_)]

        # Extract base value for Class 1 (Churn)
        exp_val = self.explainer.expected_value
        if isinstance(exp_val, (list, np.ndarray)):
            if len(exp_val) > 1:
                self.base_value = float(exp_val[1])
            else:
                self.base_value = float(exp_val[0])
        else:
            self.base_value = float(exp_val)

    def _prepare_input(self, customer_df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure required columns exist and defaults are provided for optional demographic features.
        """
        df = customer_df.copy()
        if hasattr(self.model, "feature_names_in_"):
            for col in self.model.feature_names_in_:
                if col not in df.columns:
                    if col == "Age":
                        df[col] = 45
                    elif col == "PaperlessBilling":
                        df[col] = "Yes"
                    else:
                        df[col] = 0
        return df

    def explain_instance(self, customer_df: pd.DataFrame) -> dict:
        """
        Compute local SHAP attributions for a single customer observation.

        Parameters
        ----------
        customer_df : pd.DataFrame
            Single-row DataFrame representing the customer.

        Returns
        -------
        dict
            Structured dictionary with base_value, prediction_value, ranked features,
            top positive (risk-increasing) and negative (risk-mitigating) drivers,
            and an automated human-readable narrative.
        """
        df_prepared = self._prepare_input(customer_df)
        X_trans = self.preprocessor.transform(df_prepared)

        # Compute raw SHAP values
        raw_shap = self.explainer.shap_values(X_trans)

        # Extract attributions for Class 1 (Churn)
        if isinstance(raw_shap, np.ndarray) and len(raw_shap.shape) == 3:
            shap_vec = raw_shap[0, :, 1]
        elif isinstance(raw_shap, list) and len(raw_shap) >= 2:
            shap_vec = raw_shap[1][0]
        elif isinstance(raw_shap, list) and len(raw_shap) == 1:
            shap_vec = np.array(raw_shap[0]).flatten()
        elif isinstance(raw_shap, np.ndarray):
            shap_vec = raw_shap[0] if raw_shap.ndim == 2 else raw_shap.flatten()
        else:
            shap_vec = np.array(raw_shap).flatten()

        churn_proba = float(self.model.predict_proba(df_prepared)[0][1])

        # Map post-transform features with their values and attributions
        transformed_row = X_trans[0]
        features_list = []
        for i, fname in enumerate(self.feature_names):
            s_val = float(shap_vec[i])
            val = float(transformed_row[i])
            features_list.append(
                {
                    "feature_name": fname,
                    "feature_value": val,
                    "shap_value": round(s_val, 4),
                }
            )

        # Sort descending by absolute magnitude of contribution
        features_list.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        top_risk_drivers = [f for f in features_list if f["shap_value"] > 0]
        top_retention_drivers = [f for f in features_list if f["shap_value"] < 0]

        # Construct human-readable summary narrative
        narrative_parts = []
        if top_risk_drivers:
            top_risks = [f["feature_name"] for f in top_risk_drivers[:2]]
            narrative_parts.append(f"Churn risk is primarily accelerated by {', '.join(top_risks)}")
        if top_retention_drivers:
            top_retains = [f["feature_name"] for f in top_retention_drivers[:2]]
            narrative_parts.append(
                f"partially cushioned by loyalty signals in {', '.join(top_retains)}"
            )

        if narrative_parts:
            summary_narrative = ". ".join(narrative_parts) + "."
        else:
            summary_narrative = "Account features exhibit neutral deviation from expected baseline."

        return {
            "base_value": round(self.base_value, 4),
            "prediction_value": round(churn_proba, 4),
            "churn_probability": round(churn_proba, 4),
            "features": features_list,
            "top_risk_drivers": top_risk_drivers[:5],
            "top_retention_drivers": top_retention_drivers[:5],
            "summary_narrative": summary_narrative,
        }

    def plot_waterfall(
        self,
        customer_df: pd.DataFrame,
        save_path: str | Path | None = None,
        top_n: int = 8,
    ):
        """
        Generate localized horizontal contribution plot showing positive (red)
        vs negative (green) forces pushing the score relative to the expected base value.
        """
        explanation = self.explain_instance(customer_df)
        features = explanation["features"][:top_n]
        features = list(reversed(features)) # Bottom to top for horizontal bar

        names = [f["feature_name"] for f in features]
        values = [f["shap_value"] for f in features]
        colors = ["#ef4444" if v >= 0 else "#10b981" for v in values]

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.barh(names, values, color=colors, height=0.6, edgecolor="#334155")
            ax.axvline(0, color="#64748b", linestyle="--", linewidth=1.2)

            for bar in bars:
                w = bar.get_width()
                offset = 0.005 if w >= 0 else -0.015
                ax.text(
                    w + offset,
                    bar.get_y() + bar.get_height() / 2,
                    f"{w:+.3f}",
                    va="center",
                    ha="left" if w >= 0 else "right",
                    fontsize=9,
                    weight="bold",
                    color="#0f172a",
                )

            ax.set_title(
                f"Local SHAP Feature Contributions (Base={explanation['base_value']:.2f} -> Pred={explanation['churn_probability']:.2%})",
                fontsize=12,
                weight="bold",
                pad=10,
            )
            ax.set_xlabel("SHAP Value (Contribution to Churn Probability)", fontsize=10)
            ax.grid(axis="x", alpha=0.3, linestyle=":")
            plt.tight_layout()

            if save_path is not None:
                out = Path(save_path)
                out.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(out, dpi=300)
                plt.close(fig)
                return str(out.resolve())
            return fig

        except ImportError: # pragma: no cover
            # Fallback using PIL
            from PIL import Image, ImageDraw

            img_w, img_h = 750, 420
            img = Image.new("RGB", (img_w, img_h), "#ffffff")
            draw = ImageDraw.Draw(img)

            draw.text((40, 20), "Local SHAP Risk Factor Contributions", fill="#0f172a")
            draw.text(
                (40, 42),
                f"Base={explanation['base_value']:.2f} | Churn Probability={explanation['churn_probability']:.2%}",
                fill="#475569",
            )

            center_x = 450
            start_y = 80
            row_h = 36
            scale = 800 # pixels per unit SHAP

            draw.line(
                [(center_x, start_y - 10), (center_x, start_y + len(features) * row_h)],
                fill="#94a3b8",
                width=2,
            )

            for idx, feat in enumerate(features):
                y = start_y + idx * row_h
                fname = feat["feature_name"]
                s_val = feat["shap_value"]
                bar_len = int(s_val * scale)

                draw.text((40, y + 6), fname[:32], fill="#1e293b")

                if s_val >= 0:
                    draw.rectangle([center_x, y + 4, center_x + bar_len, y + 24], fill="#ef4444")
                    draw.text((center_x + bar_len + 8, y + 6), f"+{s_val:.3f}", fill="#991b1b")
                else:
                    draw.rectangle([center_x + bar_len, y + 4, center_x, y + 24], fill="#10b981")
                    draw.text((center_x + bar_len - 50, y + 6), f"{s_val:.3f}", fill="#065f46")

            if save_path is not None:
                out = Path(save_path)
                out.parent.mkdir(parents=True, exist_ok=True)
                img.save(out)
                return str(out.resolve())
            return img
