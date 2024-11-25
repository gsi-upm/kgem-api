from pydantic import BaseModel

class Graph(BaseModel):
    id: int
    name: str
    description: str
    n_entities: int
    n_triples: int
    n_relations: int
    dataset_url: str

class Entity(BaseModel):
    name:str
    similarity:float

class Embedding(BaseModel):
    entity_name: str
    graph_name: str
    embedding: list[float]
    embedding_model: str

class Prediction(BaseModel):
    head: str
    relationship: str
    tail: str
    score: float