from fastapi import FastAPI
from api.routes import router as recommendation_router
import os
from dotenv import load_dotenv
from pathlib import Path


# env_path = Path(__file__).resolve().parent.parent / ".env"
# load_dotenv(dotenv_path=env_path)
# load_dotenv("../.env")
load_dotenv()
print("Loaded URL:", os.getenv("KGE_API_BASE")) 

app = FastAPI(title="Recommender Service")

app.include_router(recommendation_router, prefix="/recommend", tags=["Recommend"])