from fastapi import APIRouter
from pydantic_models import Prediction
from EmbeddingsLib import predict_missing_link, predict_missing_tail,predict_triplet_score
from kge_model_loader import get_model, embedding_model

router = APIRouter()



@router.get("/link/{graph_name}/{head}/{tail}")
def predict_link(graph_name:str,head:str,tail:str) -> list[Prediction]:
    '''Predicts the relationship/link between two entities (head and tail). Returns the full predicted triplet (head, relation, tail) along with its score.
    Returns a list of plausible triplets, sorted from higher to lower score.'''

    model=get_model(graph_name,embedding_model)

    relationships,scores=predict_missing_link(graph_name,model,head,tail)
    predictions=[{"head":head, "relationship":relationship, "tail": tail, "score":score} for relationship,score in zip(relationships,scores)]

    return predictions

@router.get("/entity/{graph_name}/{head}/{link}")
def predict_entity(graph_name:str,head:str,relationship:str) -> list[Prediction]:
    '''Predicts the missing tail entity of a triplet given a head entity and a relation. Return the full predicted triplet (head, relation, tail) along with its score.
    Returns a list of plausible triplets, sorted from higher to lower score.'''

    model=get_model(graph_name,embedding_model)

    tails,scores=predict_missing_tail(graph_name,model,head,relationship)
    predictions=[{"head":head, "relationship":relationship, "tail": tail, "score":score} for tail,score in zip(tails,scores)]
    return predictions

@router.get("/probability/{graph_name}/{head}/{relationship}/{tail}")
def get_triplet_score(graph_name,head,relationship,tail) -> Prediction:
    '''Gets the score (plausibility) of a specific triplet formed by "head", "relationship", and "tail".'''
    
    model=get_model(graph_name,embedding_model)

    score=predict_triplet_score(graph_name,model,head,relationship,tail)
    return {"head":head, "relationship":relationship, "tail": tail, "score":score}