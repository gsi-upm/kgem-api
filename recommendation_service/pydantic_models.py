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
    graph: str
    embedding_model: str

class RecommendationResponse(BaseModel):
    relevance_score: Union[float, int]
    model_used: ModelEnum
    mode: ModeEnum
    graph: str
    embedding_model: str
    used_entities: dict[str, list[str]]  # Dictionary with keys "cluster 1" and "cluster 2" containing lists of entities
    features: dict[str,float]
    
