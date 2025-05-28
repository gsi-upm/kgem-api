from pykeen.datasets import get_dataset
from pykeen import predict
from pykeen.triples import TriplesFactory

import torch

import torch
from typing import List

import pykeen.nn

import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from scipy.optimize import minimize
from scipy.spatial.distance import cdist

from fastapi import HTTPException

from pprint import pprint



# embedding functions
def calculate_embeddings(model,entity:str):
    ''' '''
    try:        
        # entity_id = torch.as_tensor(triples_factory.entities_to_ids([entity]))
        entity_id = torch.as_tensor(model.triples_factory.entities_to_ids([entity]))
                
        entity_representation_modules: List['pykeen.nn.Representation'] = model.entity_representations
        entity_embeddings: pykeen.nn.Embedding = entity_representation_modules[0]
        entity_embedding_tensor: torch.FloatTensor = entity_embeddings(indices=entity_id).detach()[0]
        
        return entity_embedding_tensor
    
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Entity '{entity}' not found in graph.")

def calculate_entity_from_embedding(model:str, target_embedding, k=10):
    #OBTENER VETOR CON LA REPRESENTACIÓN DE TODAS LAS ENTIDADES
    triples_factory = model.triples_factory

    entity_representation_modules: List['pykeen.nn.Representation'] = model.entity_representations
    entity_embeddings: pykeen.nn.Embedding = entity_representation_modules[0]
    entity_embedding_tensor: torch.FloatTensor = entity_embeddings()

    target_embedding=torch.tensor([float(i) for i in target_embedding]) # convertir los datos de la api
    #CALCULAR SIMILITUD DE COSENO (probar otras opciones para calcular similitud por si hay entidades muy parecidas, intervalos de confianza, etc)
    try:
        similarities = F.cosine_similarity(target_embedding.unsqueeze(0), entity_embedding_tensor.cpu(), dim=1)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    #most_similar_index = torch.argmax(similarities).item()
    top_k_similarities, most_similar_indices = torch.topk(similarities, k=k)
    #most_similar_index=most_similar_indices[0]
    most_similar_indices=most_similar_indices.numpy()
    #print("Target:",target_embedding)
    #print("Embeddings list:",entity_embedding_tensor)
    #print("Similarities:",similarities)
    #print("Target embedding ID:", most_similar_index)

    target_labels=[triples_factory.entity_id_to_label[idx] for idx in most_similar_indices]
    #print("Target labels:",target_labels)
    #print("Similarities:",top_k_similarities)
    return target_labels,top_k_similarities


# predictions
def predict_missing_link(model, head, tail, k=5):

    triples_factory = model.triples_factory
    
    try:
        df = predict.predict_target(
        model=model,
        head=head,
        tail=tail,
        triples_factory=triples_factory)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"Either '{head}' or '{tail}' not found in graph.\nAdditional Exception info:{e}")

    labels = df.df['relation_label'].tolist()[:k]
    scores = df.df['score'].tolist()[:k]
    
    return labels, scores

def predict_missing_tail(model, head, relationship, k=5):

    triples_factory = model.triples_factory
    
    try:
        df = predict.predict_target(
        model=model,
        head=head,
        relation=relationship,
        triples_factory=triples_factory)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"Either Entity '{head}' or link '{relationship}' not found in graph.\nAdditional Exception info:{e}")

    labels = df.df['tail_label'].tolist()[:k]
    scores = df.df['score'].tolist()[:k]
    
    return labels, scores

def predict_triplet_score(model, head, relationship,tail):

    triples_factory = model.triples_factory
    
    try:
        prediction = predict.predict_triples(
        model=model,
        triples_factory=triples_factory,
        triples=(head,relationship,tail),batch_size=1)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Either Entity '{head}' or Link '{relationship}' not found in graph.")

    return prediction.scores[0]


# similarity functions
def entity_cosine_similarity(model,entity1,entity2):

    embedding1=calculate_embeddings(model,entity1)
    embedding2=calculate_embeddings(model,entity2)

    similarity = F.cosine_similarity(embedding1, embedding2, dim=0)

    return similarity.item()

def multiple_entity_cosine_similarity(model,entity_list1,entity_list2,similarity_metric="mean"):
    
    similarity_matrix = np.zeros((len(entity_list1), len(entity_list2)))

    for i in range(len(entity_list1)):
        for j in range(len(entity_list2)):
            similarity_matrix[i, j] = entity_cosine_similarity(model,entity_list1[i],entity_list2[j])
    
    match similarity_metric:
        case "mean":
            similarity = np.mean(similarity_matrix)
        case "min":
            similarity = np.min(similarity_matrix)
        case "max":
            similarity = np.max(similarity_matrix)
        case "median":
            similarity = np.median(similarity_matrix)
        case "sum":
            similarity = np.sum(similarity_matrix)
        case _:
            raise HTTPException(status_code=400, detail=f"Invalid similarity metric '{similarity_metric}'. Allowed values are: mean, min, max, median, sum.")        

    return similarity

def embeddings_cosine_similarity(embedding1,embedding2):

    embedding1=torch.tensor([float(i) for i in embedding1])
    embedding2=torch.tensor([float(i) for i in embedding2])

    try:
        similarity = F.cosine_similarity(embedding1, embedding2, dim=0)
        return similarity
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    

def multiple_embedding_cosine_similarity(embeddings1,embeddings2,similarity_metric="mean"):
    
    embeddings1=[torch.tensor(t) for t in embeddings1]
    embeddings2=[torch.tensor(t) for t in embeddings2]

    similarity_matrix = np.zeros((len(embeddings1), len(embeddings2)))

    for i in range(len(embeddings1)):
        for j in range(len(embeddings2)):
            similarity_matrix[i, j] = embeddings_cosine_similarity(embeddings1[i],embeddings2[j])
    
    match similarity_metric:
        case "mean":
            similarity = np.mean(similarity_matrix)
        case "min":
            similarity = np.min(similarity_matrix)
        case "max":
            similarity = np.max(similarity_matrix)
        case "median":
            similarity = np.median(similarity_matrix)
        case "sum":
            similarity = np.sum(similarity_matrix)
        case _:
            raise HTTPException(status_code=400, detail=f"Invalid similarity metric '{similarity_metric}'. Allowed values are: mean, min, max, median, sum.") 

    return similarity


# overlapping and center operations
def calculate_geometric_median(embeddings):
    # centro geométrico
    eps = torch.finfo(embeddings.dtype).tiny
    embeddings = torch.clamp(embeddings, min=eps)  # Avoid zero values
    center = torch.prod(embeddings, axis=0) ** (1 / len(embeddings))
    return center

def calculate_centroid(embeddings):
    #media aritmética: coordenadas resultantes de hacer la media de todas las entidades
    centroid = torch.mean(embeddings,axis=0)
    return centroid

def calculate_radius(embeddings,center,radius_type="mean"):
    """Computes the radius of the hypershpere formed by the entity emebddings.
    Supporteed values: mean (mean of the distances from the center to each entity), median, max"""
    match radius_type:
        case "max":
            radius = torch.max(torch.norm(embeddings - center, dim=1)).item()
        case "mean":
            radius = torch.mean(torch.norm(embeddings - center, dim=1)).item()
        case "median":
            radius = torch.median(torch.norm(embeddings - center, dim=1)).item()
        case _:
            raise HTTPException(status_code=400, detail=f"Invalid radius type metric '{radius_type}'. Allowed values are: mean, median, max.") 
        
    return radius

def calculate_centers(embeddings, center_type="all"):
    '''If center_type == all returns all centers.
    implemented centers: centroid, geometric.'''    
    
    embeddings = torch.stack(embeddings) # stack the list of embeddings to a single tensor

    centers = {}

    if center_type in {"geometric_median", "all"}:
        geometric_median = calculate_geometric_median(embeddings=embeddings)
        r_median = calculate_radius(embeddings,geometric_median)
        centers["geometric_median"] = {"point": torch.tensor(geometric_median), 
                                       "radii": {"median":r_median, # con el centro geométrico el radio es el mismo en los tres casos porque debería estar a la misma distancia de todas las entidades
                                                 "mean":r_median,
                                                 "max":r_median}}

    if center_type in {"centroid", "all"}:
        centroid = calculate_centroid(embeddings=embeddings)
        centers["centroid"] = {"point": torch.tensor(centroid), 
                               "radii": {"median":calculate_radius(embeddings, centroid, radius_type="median"),
                                            "mean":calculate_radius(embeddings, centroid, radius_type="mean"),
                                            "max":calculate_radius(embeddings, centroid, radius_type="max")}}

    if center_type in centers:
        return {center_type: centers[center_type]}
    elif center_type == "all":
        return centers
    else:
        raise HTTPException(status_code=400, detail=f"Unrecognized center_type: '{center_type}'.\nAllowed values are: 'all', 'centroid', 'geometric_median'.") 

def compute_center_distance(center1, center2):
    """Computes the euclidean distance between two points in the embedding space."""
    center1 = torch.tensor(center1)
    center2 = torch.tensor(center2)
    
    distance = torch.norm(center1 - center2).item()
    
    return distance
    
def calculate_overlap(center1, center2, radius1, radius2):
    # if one cluster is a single point and the other is not
    if radius1 == 0 and radius2 > 0:
        distance = compute_center_distance(center2, center1)
        if distance <= radius2:
            return 1.0 # return maximum overlap if the entity falls within the cluster range
        else:
            return 0.0 # return 0 overlap if the entity does not fall within the cluster range
    elif radius2 == 0 and radius1 > 0:
        distance =compute_center_distance(center1, center2)
        if distance <= radius1:
            return 1.0
        else:
            return 0.0
    
    # distance between the centers of the circles
    distance = compute_center_distance(center1, center2)

    # if both clusters are single points
    if radius1 == radius2 == 0:
        if distance == radius1 == radius2 == 0:
            return 1.0 # overlap is maximum because the same entity appears in both news
        else:
            return 0.0 # two different entities (no overlapping)

    # clusters with more than one entity
    if distance >= radius1 + radius2: # if the distance is bigger than both radius there is no overlap
        return 0.0

    # caso en el que un círculo está completamente dentro del otro: El solape es total.
    elif distance <= abs(radius1 - radius2): 
        return 1.0               # intersección = círculo pequeño completo

    # OVERLAPPING RATIO: generalización del cálculo del area del círculo
    else:
        # convertir radios a tensores para usar en torch.acos y torch.sqrt
        radius1 = torch.tensor(radius1, dtype=torch.float)
        radius2 = torch.tensor(radius2, dtype=torch.float)
        
        overlap = (
            radius1**2 * torch.acos((distance**2 + radius1**2 - radius2**2) /
                                    (2 * distance * radius1))
          + radius2**2 * torch.acos((distance**2 + radius2**2 - radius1**2) /
                                    (2 * distance * radius2))
          - 0.5 * torch.sqrt(
                (-distance + radius1 + radius2)
              * ( distance + radius1 - radius2)
              * ( distance - radius1 + radius2)
              * ( distance + radius1 + radius2)
            )
        )
        
        return float(overlap / (torch.pi * min(radius1**2, radius2**2))) # dividimos por el área del círculo más pequeño
if __name__ == "__main__": 
    pass