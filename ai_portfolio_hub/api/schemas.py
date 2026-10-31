"""
Pydantic v2 Request & Response Data Contracts for Unified Multi-Domain AI Gateway.

Provides strict serialization, type validation, and documentation schemas across:
- Tabular Churn Inference & SHAP
- Vision CNN Classification & Grad-CAM
- NLP Transformer Contextual Sentiment Analysis
- RL PER-D3QN CartPole Trajectory Playback
"""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ==========================================
# 1. Tabular Churn Schemas
# ==========================================
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


class TabularInput(BaseModel):
    Tenure_Months: int = Field(..., ge=0, le=120, description="Customer tenure in months", examples=[12])
    MonthlyCharges: float = Field(..., ge=0.0, le=1000.0, description="Monthly recurring charge in USD", examples=[85.0])
    TotalTickets: int = Field(..., ge=0, le=50, description="Support inquiries filed", examples=[3])
    ContractType: ContractTypeEnum = Field(..., description="Billing agreement term", examples=[ContractTypeEnum.month_to_month])
    TechSupport: TechSupportEnum = Field(..., description="Technical support add-on status", examples=[TechSupportEnum.no])
    PaymentMethod: PaymentMethodEnum = Field(..., description="Payment billing channel", examples=[PaymentMethodEnum.electronic_check])
    Age: Optional[int] = Field(default=45, ge=18, le=120, description="Customer age")
    PaperlessBilling: Optional[str] = Field(default="Yes", description="Electronic statement option")


class TabularOutput(BaseModel):
    churn_probability: float = Field(..., description="Predicted churn likelihood [0.0 - 1.0]")
    risk_level: str = Field(..., description="Risk tier: Low, Moderate, High")
    recommended_action: str = Field(..., description="Actionable retention intervention strategy")
    shap_values: Dict[str, float] = Field(..., description="Local SHAP feature attributions")
    base_value: float = Field(..., description="Expected baseline log-odds / probability")


# ==========================================
# 2. Vision CNN & Grad-CAM Schemas
# ==========================================
class VisionPredictionItem(BaseModel):
    rank: int
    label: str
    probability: float
    percentage: str


class VisionOutput(BaseModel):
    top_class: str
    top_confidence: float
    predictions: List[VisionPredictionItem]
    gradcam_base64: str = Field(..., description="Superimposed Grad-CAM saliency overlay as Base64 data URI")
    image_shape: List[int]


# ==========================================
# 3. NLP Transformer Schemas
# ==========================================
class NLPInput(BaseModel):
    text: str = Field(..., min_length=2, max_length=1000, description="Text review or feedback to analyze")


class NLPOutput(BaseModel):
    label: str = Field(..., description="Sentiment classification: POSITIVE or NEGATIVE")
    confidence: float = Field(..., description="Probability of predicted class [0.0 - 1.0]")
    probabilities: Dict[str, float] = Field(..., description="Softmax probabilities across all classes")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")


# ==========================================
# 4. Reinforcement Learning Schemas
# ==========================================
class RLPlaybackRequest(BaseModel):
    episodes: int = Field(default=1, ge=1, le=5, description="Number of evaluation simulation runs")
    seed: int = Field(default=42, description="Random seed for environment reset")


class RLPlaybackResponse(BaseModel):
    total_reward: float
    steps: int
    state_history: List[List[float]] = Field(..., description="Step-by-step 4D state vectors [x, x_dot, theta, theta_dot]")
    q_values_history: List[List[float]] = Field(..., description="Step-by-step Q-values [Q(left), Q(right)]")
    actions_taken: List[int] = Field(..., description="Selected actions (0=left, 1=right)")
    success: bool = Field(..., description="Whether pole stayed balanced through maximum timesteps (500)")


# ==========================================
# 5. Global Health & Telemetry Schemas
# ==========================================
class SystemHealthResponse(BaseModel):
    status: str
    device: str
    models_loaded: Dict[str, bool]
    uptime_seconds: float

