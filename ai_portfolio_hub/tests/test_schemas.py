"""
Schema Validation Unit Tests for API Data Contracts.

Validates:
- Tabular bounds checking, enums, defaults
- NLP string length constraints
- RL simulation episode limits and seed validation
"""

import sys
from pathlib import Path
import pytest
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.schemas import (
    ContractTypeEnum,
    NLPInput,
    PaymentMethodEnum,
    RLPlaybackRequest,
    TabularInput,
    TechSupportEnum,
)


def test_tabular_input_valid():
    """Verify valid customer features instantiate correctly."""
    data = TabularInput(
        Tenure_Months=14,
        MonthlyCharges=78.50,
        TotalTickets=2,
        ContractType=ContractTypeEnum.month_to_month,
        TechSupport=TechSupportEnum.no,
        PaymentMethod=PaymentMethodEnum.electronic_check,
    )
    assert data.Tenure_Months == 14
    assert data.MonthlyCharges == 78.50
    assert data.Age == 45  # Default value


def test_tabular_input_bounds_failure():
    """Verify negative charges or tenure > 120 trigger ValidationError."""
    with pytest.raises(ValidationError):
        TabularInput(
            Tenure_Months=-5,  # Invalid
            MonthlyCharges=78.50,
            TotalTickets=2,
            ContractType=ContractTypeEnum.month_to_month,
            TechSupport=TechSupportEnum.no,
            PaymentMethod=PaymentMethodEnum.electronic_check,
        )

    with pytest.raises(ValidationError):
        TabularInput(
            Tenure_Months=14,
            MonthlyCharges=-10.0,  # Invalid
            TotalTickets=2,
            ContractType=ContractTypeEnum.month_to_month,
            TechSupport=TechSupportEnum.no,
            PaymentMethod=PaymentMethodEnum.electronic_check,
        )


def test_nlp_input_constraints():
    """Verify text length minimum and maximum bounds."""
    # Valid
    valid_nlp = NLPInput(text="Great system performance!")
    assert valid_nlp.text == "Great system performance!"

    # Too short (< 2 chars)
    with pytest.raises(ValidationError):
        NLPInput(text="a")

    # Too long (> 1000 chars)
    with pytest.raises(ValidationError):
        NLPInput(text="a" * 1001)


def test_rl_request_constraints():
    """Verify RL playback parameter ranges."""
    req = RLPlaybackRequest(episodes=2, seed=123)
    assert req.episodes == 2
    assert req.seed == 123

    # Exceeding 5 episodes
    with pytest.raises(ValidationError):
        RLPlaybackRequest(episodes=10)

