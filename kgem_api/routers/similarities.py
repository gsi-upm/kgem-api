from fastapi import APIRouter
from EmbeddingsLib import calculate_embeddings,calculate_entity_from_embedding,entity_cosine_similarity, embeddings_cosine_similarity,multiple_entity_cosine_similarity, multiple_embedding_cosine_similarity
from kge_model_loader import get_model
from pydantic_models import Entity, CosineSimilarityResponse

router = APIRouter()


@router.get("/closest-entities/{graph_name}/{embedding_model}/{entity}")
def get_closest_entities(graph_name:str,embedding_model:str,entity:str,k:int=5) -> list[Entity]:
    '''Returns a list of the k closest entities to the specified entity based on cosine similarity.'''

    model=get_model(graph_name,embedding_model)

    embedding=calculate_embeddings(model,entity)

    entities,similarities=calculate_entity_from_embedding(model,embedding,k=k)

    entities=[{"name":label,"similarity":similarity.item()} for (label,similarity) in zip(entities,list(similarities))]

    return entities

@router.get("/cosine-similarity/entities/{graph_name}/{embedding_model}/{entity1}/{entity2}", response_model=CosineSimilarityResponse)
def get_cosine_similarity_from_entities(graph_name:str,embedding_model:str,entity1:str,entity2:str):
    '''Computes the cosine similarity between two entities in the specified graph.'''

    model=get_model(graph_name,embedding_model)
    similarity=entity_cosine_similarity(model,entity1,entity2)

    return {"similarity":similarity}

@router.post("/cosine-similarity/embeddings", response_model=CosineSimilarityResponse)
def get_cosine_similarity_from_embeddings(embedding_1:list[float],embedding_2:list[float]):
    '''Calculates the cosine similarity between two embedding vectors provided in the request body.'''

    similarity=embeddings_cosine_similarity(embedding_1,embedding_2)

    return {"similarity":similarity}

@router.post("/cosine-similarity/entities/multiple/{graph_name}/{embedding_model}/{metric}", response_model=CosineSimilarityResponse)
def get_cosine_similarity_multiple_entities(graph_name:str,embedding_model:str,entity_list1:list[str],entity_list2:list[str],metric:str,raise_on_missing:bool=True):
    '''Computes cosine similarity between all pairs of entities from two lists of entities specified in the request body.
    The pairwise similarities are then aggregated using the specified metric.
    
    available metrics: mean,min,max,median,sum'''

    model=get_model(graph_name,embedding_model)
    similarity=multiple_entity_cosine_similarity(model,entity_list1,entity_list2,metric,raise_on_missing)

    return {"similarity":similarity, "metric_used":metric}

@router.post("/cosine-similarity/embeddings/multiple/{metric}", response_model=CosineSimilarityResponse)
def get_cosine_similarity_multiple_embeddings(embedding_list1:list[float],embedding_list2:list[float],metric:str):
    '''Calculates the cosine similarity between two groups of embedding vectors. The similarity scores are aggregated 
    using the specified metric.
    raise_on_missing: if True, raises an error if the entity is not within the graph. If False, it performs the aggregation ignoring all missing entities.
    available metrics: mean,min,max,median,sum'''

    similarity=multiple_embedding_cosine_similarity(embedding_list1,embedding_list2,metric)

    return {"similarity":similarity, "metric used":metric}