"""
Pydantic schemas for the prediction API.

These define the exact shape of incoming requests and outgoing responses.
Distinct from contracts/schemas.py (which is the DE↔DS internal contract);
this is the external HTTP contract for API consumers.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class FlowFeatures(BaseModel):
    """One preprocessed flow, ready for model input.

    Matches the 77 feature columns in the training splits (post-preprocessing).
    Serialized as JSON with feature names as keys. For a real production API,
    we'd likely accept raw pre-preprocessing features and run the pipeline
    server-side, but for now we accept model-ready features.
    """
    # Accept arbitrary feature dict — full validation would list all 77.
    # This is simpler and more resilient to feature-set changes.
    features: dict[str, float] = Field(
        ...,
        description="Feature name → value. Must match model's training features.",
    )


class TopFeature(BaseModel):
    """One feature's contribution to a prediction (SHAP-derived)."""
    feature: str
    value: float
    shap_contribution: float


class PredictionResponse(BaseModel):
    predicted_class: str
    probability: float
    all_probabilities: dict[str, float]
    top_features: Optional[list[TopFeature]] = None
    model_version: str
    schema_version: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_alias: str
    schema_version: str
    n_features: int
    classes: list[str]