from pydantic import BaseModel
from typing import Union
from enum import Enum

class ModeEnum(str, Enum):
    classification = "classification"
    regression = "regression"

class ModelEnum(str, Enum):
    random_forest = "random_forest"
    
class RecommendationRequest(BaseModel):
    news1_entities: list[str]
    news2_entities: list[str]
    mode: ModeEnum
    model_name: ModelEnum

class RecommendationResponse(BaseModel):
    relevance_score: Union[float, int]
    model_used: ModelEnum
    mode: ModeEnum
    features: dict[str,float]
