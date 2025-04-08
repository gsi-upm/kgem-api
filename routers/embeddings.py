from fastapi import APIRouter,Body
from pydantic_models import Embedding, Entity
from EmbeddingsLib import calculate_embeddings,calculate_entity_from_embedding
from kge_model_loader import get_model
from graph_loading_utils import get_available_embedding_models
router = APIRouter()


@router.get("/")
def list_embedding_models() -> list[str]:
    '''Retrieves a list of all available embedding models.'''
    return get_available_embedding_models()

@router.get("/by-entity/{graph_name}/{embedding_model}/{entity}") 
def embedding_from_entity(graph_name: str, embedding_model:str, entity: str) -> Embedding:
    """
    Retrieves the embedding vector for a given entity name within a specified graph."""
    model = get_model(graph_name,embedding_model)
    embedding = calculate_embeddings(graph_name,model, entity)

    #transformar en una lista de numeros decimales para poder usarlos en la API
    embedding=[float(i) for i in embedding]

    return Embedding(entity_name=entity, graph_name=graph_name, embedding=embedding, embedding_model=embedding_model)


@router.post("/closest-entity/{graph_name}/{embedding_model}", response_model=list[Entity])
def entity_from_embedding(graph_name: str,
                          embedding_model:str,
                          embedding:list[float],
                          k: int)-> list[Entity]:
    '''Find the k closests entities to a given embedding vector within a specified graph.'''
    
    model = get_model(graph_name,embedding_model)

    entities,similarities=calculate_entity_from_embedding(graph_name,model,embedding,k=k)
    entities=[{"name":label,"similarity":similarity} for label,similarity in zip(entities,similarities)]

    return entities