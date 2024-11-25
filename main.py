from fastapi import FastAPI
from EmbeddingsLib import *
from pydantic_models import *
from routers import graphs, embeddings, predictions, similarities,centers


app = FastAPI()

app.include_router(graphs.router, prefix="/graphs", tags=["Graphs"])
app.include_router(embeddings.router, prefix="/embeddings", tags=["Embeddings"])
app.include_router(predictions.router, prefix="/predictions", tags=["Predictions"])
app.include_router(similarities.router, prefix="/similarities", tags=["Similarities"])
app.include_router(centers.router, prefix="/centers", tags=["Centers"])




@app.get("/")
def read_root():
    '''Root endpoint.'''
    return {"test message": "hello world"}



