from fastapi import APIRouter,Body
from pydantic_models import Embedding, Entity
from EmbeddingsLib import calculate_embeddings,calculate_entity_from_embedding
from kge_model_loader import get_model, embedding_model

router = APIRouter()

@router.get("/by-entity/{graph_name}/{entity}") 
def embedding_from_entity(graph_name: str, entity: str) -> Embedding:
    """
    Retrieves the embedding vector for a given entity name within a specified graph."""
    model = get_model(graph_name,embedding_model)
    embedding = calculate_embeddings(graph_name,model, entity)

    #transformar en una lista de numeros decimales para poder usarlos en la API
    embedding=[float(i) for i in embedding]

    return Embedding(entity_name=entity, graph_name=graph_name, embedding=embedding, embedding_model=embedding_model)


@router.post("/closest-entity/{graph_name}")
def entity_from_embedding(graph_name: str,
                          embedding:list[float])-> list[Entity]:
    '''Find the closest entity to a given embedding vector within a specified graph.'''
    
    model = get_model(graph_name,embedding_model)

    entities,similarities=calculate_entity_from_embedding(graph_name,model,embedding)
    entities=[{"name":label,"similarity":similarity} for label,similarity in zip(entities,similarities)]

    return entities