"""
Pydantic v2 Request & Response Data Contracts.

Defines strict input validation schemas, enumeration constraints, structured
inference & SHAP explanation response formats, and retention alerting toggles.
"""

import uuid
from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class ContractTypeEnum(str, Enum):
    month_to_month = "Month-to-month"
    one_year = "One year"
    two_year = "Two year"


class TechSupportEnum(str, Enum):
    yes = "Yes"
    no = "No"


class PaymentMethodEnum(str, Enum):
    electronic_check = "Electronic check"
    mailed_check = "Mailed check"
    bank_transfer = "Bank transfer"
    credit_card = "Credit card"


class CustomerFeatures(BaseModel):
    Customer_ID: Optional[str] = Field(
        default_factory=lambda: f"CUST-{uuid.uuid4().hex[:8].upper()}",
        description="Unique customer account identifier",
        examples=["CUST-1001"],
    )
    Tenure_Months: int = Field(
        ...,
        ge=0,
        le=120,
        description="Customer subscription tenure in months",
        examples=[14],
    )
    MonthlyCharges: float = Field(
        ...,
        ge=0.0,
        le=1000.0,
        description="Monthly subscription bill in USD",
        examples=[78.50],
    )
    TotalTickets: int = Field(
        ...,
        ge=0,
        le=50,
        description="Total support and service tickets filed",
        examples=[3],
    )
    ContractType: ContractTypeEnum = Field(
        ...,
        description="Billing agreement term",
        examples=[ContractTypeEnum.month_to_month],
    )
    TechSupport: TechSupportEnum = Field(
        ...,
        description="Whether customer has technical support add-on",
        examples=[TechSupportEnum.no],
    )
    PaymentMethod: PaymentMethodEnum = Field(
        ...,
        description="Customer billing payment method",
        examples=[PaymentMethodEnum.electronic_check],
    )
    Age: Optional[int] = Field(
        default=45,
        ge=0,
        le=120,
        description="Customer age in years",
        examples=[42],
    )
    PaperlessBilling: Optional[str] = Field(
        default="Yes",
        description="Whether customer opted for digital paperless invoicing",
        examples=["Yes"],
    )
    notify_retention_team: bool = Field(
        default=True,
        description="Whether to dispatch proactive webhook alerts if account exhibits high churn risk",
    )

    def to_feature_dict(self) -> dict:
        """
        Convert validated features to a dictionary suitable for pandas DataFrame ML ingestion.
        Excludes metadata identifiers and operational alert flags.
        """
        data = {
            "Tenure_Months": self.Tenure_Months,
            "MonthlyCharges": self.MonthlyCharges,
            "TotalTickets": self.TotalTickets,
            "ContractType": self.ContractType.value,
            "TechSupport": self.TechSupport.value,
            "PaymentMethod": self.PaymentMethod.value,
        }
        if self.Age is not None:
            data["Age"] = self.Age
        if self.PaperlessBilling is not None:
            data["PaperlessBilling"] = self.PaperlessBilling
        return data


class ChurnPredictionResponse(BaseModel):
    Customer_ID: Optional[str] = Field(
        default=None,
        description="Customer account identifier",
        examples=["CUST-1001"],
    )
    churn_prediction: int = Field(
        ...,
        description="Binary classification outcome: 0 = Retained, 1 = Churned",
        examples=[1],
    )
    churn_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence probability for churn (0.0 to 1.0)",
        examples=[0.8245],
    )
    risk_level: str = Field(
        ...,
        description="Operational risk category: Low, Medium, High",
        examples=["High"],
    )
    recommended_action: str = Field(
        ...,
        description="Prescribed customer retention intervention policy",
        examples=["Offer targeted loyalty discount and schedule priority retention callback."],
    )
    alert_dispatched: bool = Field(
        default=False,
        description="Whether an automated retention alert was scheduled for dispatch",
        examples=[True],
    )


class BatchCustomerInput(BaseModel):
    customers: List[CustomerFeatures] = Field(
        ...,
        description="List of customer feature payloads for batch processing",
    )


class BatchPredictionResponse(BaseModel):
    total_processed: int = Field(
        ...,
        description="Total customer records processed in batch",
        examples=[2],
    )
    predictions: List[ChurnPredictionResponse] = Field(
        ...,
        description="List of structured prediction responses",
    )


class FeatureContribution(BaseModel):
    feature_name: str = Field(
        ...,
        description="Transformed feature column name",
        examples=["Tenure_Months"],
    )
    feature_value: Any = Field(
        ...,
        description="Standardized or encoded value of the feature",
        examples=[-1.52],
    )
    shap_value: float = Field(
        ...,
        description="SHAP attribution value (positive increases churn risk, negative decreases)",
        examples=[0.145],
    )


class ExplanationResponse(BaseModel):
    churn_probability: float = Field(
        ...,
        description="Predicted probability of churn",
        examples=[0.9299],
    )
    base_value: float = Field(
        ...,
        description="SHAP expected baseline probability across training population",
        examples=[0.5008],
    )
    top_risk_drivers: List[FeatureContribution] = Field(
        ...,
        description="Top features pushing customer toward attrition (positive SHAP values)",
    )
    top_retention_drivers: List[FeatureContribution] = Field(
        ...,
        description="Top features mitigating customer churn (negative SHAP values)",
    )
    summary_narrative: str = Field(
        ...,
        description="Automated plain-English explanation of risk factors",
        examples=["Churn risk is primarily accelerated by Tenure_Months, MonthlyCharges."],
    )
