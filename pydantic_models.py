from pydantic import BaseModel, Field
from typing import List, Optional


class Graph(BaseModel):
    """
    Schema representing an available Knowledge Graph with its metadata.
    """
    name: str = Field(
        ..., 
        description="The verbous name of the graph.",
        example="SampleGraph"
    )
    description: str = Field(
        ..., 
        description="A brief description of the graph.",
        example="A sample knowledge graph for demonstration purposes."
    )
    n_entities: int = Field(
        ..., 
        description="The number of entities in the graph.",
        example=1000
    )
    n_triples: int = Field(
        ..., 
        description="The number of triples (relationships) in the graph.",
        example=5000
    )
    n_relations: int = Field(
        ..., 
        description="The number of unique relationships in the graph.",
        example=50
    )
    dataset_url: str = Field(
        ..., 
        description="The URL where the graph dataset can be downloaded.",
        example="https://example.com/dataset/graph1"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "name": "Nations",
                "description": "Graph representation of relationships between countries.",
                "n_entities": 14,
                "n_triples": 1992,
                "n_relations": 55,
                "dataset_url": "https://example.com/dataset/nations"
            }
        }


class Entity(BaseModel):
    """
    Schema representing a response with an entity and its similarity score.
    """
    name: str = Field(
        ..., 
        description="The name of the entity.",
        example="Entity1"
    )
    similarity: Optional[float] = Field(
        ..., 
        description="The similarity score of the entity, ranging from 0 to 1.",
        example=0.85
    )

    class Config:
        json_schema_extra = {
            "example": {
                "name": "china",
                "similarity": 0.85
            }
        }


class Embedding(BaseModel):
    """
    Schema representing an embedding for an entity in a graph.
    """
    entity_name: str = Field(
        ..., 
        description="The name of the entity associated with the embedding.",
        example="Entity1"
    )
    graph_name: str = Field(
        ..., 
        description="The name of the graph to which the entity belongs.",
        example="SampleGraph"
    )
    embedding: list[float] = Field(
        ..., 
        description="A list of float values representing the entity's embedding.",
        example=[0.12, 0.53, 0.78, 0.45]
    )
    embedding_model: str = Field(
        ..., 
        description="The name of the embedding model used.",
        example="TransE"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "entity_name": "china",
                "graph_name": "nations",
                "embedding": [0.12, 0.53, 0.78, 0.45],
                "embedding_model": "TransE"
            }
        }


class Prediction(BaseModel):
    """
    Schema representing a prediction for a triple in the graph.
    """
    head: str = Field(
        ..., 
        description="The head entity in the predicted triple.",
        example="Entity1"
    )
    relationship: str = Field(
        ..., 
        description="The relationship or predicate between the head and tail entities.",
        example="related_to"
    )
    tail: str = Field(
        ..., 
        description="The tail entity in the predicted triple.",
        example="Entity2"
    )
    score: float = Field(
        ..., 
        description="The value of the model's score function for the prediction.",
        example=0.95
    )

    class Config:
        json_schema_extra = {
            "example": {
                "head": "china",
                "relationship": "exports",
                "tail": "usa",
                "score": 0.95
            }
        }

class CosineSimilarityResponse(BaseModel):
    similarity: float = Field(..., description="The computed cosine similarity score.")
    metric_used: Optional[str] = Field(None, description="The metric used for aggregation, if applicable.")
    class Config:
        json_schema_extra = {
            "example": {
                "similarity": 0.85,
                "metric_used": "average",
            }
        }