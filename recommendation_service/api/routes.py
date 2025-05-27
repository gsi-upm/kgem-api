from fastapi import APIRouter
from core.recommendation import predict
from pydantic_models import RecommendationRequest, RecommendationResponse

router = APIRouter()

@router.post("/", response_model=RecommendationResponse)
def recommend(data: RecommendationRequest):
    return predict(data)
