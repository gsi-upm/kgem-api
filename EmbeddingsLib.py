from pykeen.datasets import get_dataset
from pykeen import predict

import torch
from typing import List

import pykeen.nn

import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from scipy.optimize import minimize
from scipy.spatial.distance import cdist


# embedding functions
def calculate_embeddings(dataset,model,entity:str):
    ''' '''

    triples_factory = get_dataset(dataset=dataset).training
    entity_id = torch.as_tensor(triples_factory.entities_to_ids([entity]))

    entity_representation_modules: List['pykeen.nn.Representation'] = model.entity_representations
    entity_embeddings: pykeen.nn.Embedding = entity_representation_modules[0]
    entity_embedding_tensor: torch.FloatTensor = entity_embeddings(indices=entity_id).detach()[0]
    
    return entity_embedding_tensor

def calculate_entity_from_embedding(dataset:str, model:str, target_embedding, k=10):

    #OBTENER VETOR CON LA REPRESENTACIÓN DE TODAS LAS ENTIDADES
    triples_factory = get_dataset(dataset=dataset).training

    entity_representation_modules: List['pykeen.nn.Representation'] = model.entity_representations
    entity_embeddings: pykeen.nn.Embedding = entity_representation_modules[0]
    entity_embedding_tensor: torch.FloatTensor = entity_embeddings()

    target_embedding=torch.tensor([float(i) for i in target_embedding]) # convertir los datos de la api
    #CALCULAR SIMILITUD DE COSENO (probar otras opciones para calcular similitud por is hay entidades muy parecidas, intervalos de confianza, etc)

    similarities = F.cosine_similarity(target_embedding.unsqueeze(0), entity_embedding_tensor, dim=1)
    #most_similar_index = torch.argmax(similarities).item()
    top_k_similarities, most_similar_indices = torch.topk(similarities, k=k)
    #most_similar_index=most_similar_indices[0]
    most_similar_indices=most_similar_indices.numpy()
    #print("Target:",target_embedding)
    #print("Embeddings list:",entity_embedding_tensor)
    #print("Similarities:",similarities)
    #print("Target embedding ID:", most_similar_index)

    target_labels=[triples_factory.entity_id_to_label[idx] for idx in most_similar_indices]
    print("Target labels:",target_labels)
    print("Similarities:",top_k_similarities)
    return target_labels,top_k_similarities


# predictions
def predict_missing_link(dataset:str, model, head, tail, k=5):

    triples_factory = get_dataset(dataset=dataset).training

    df = predict.predict_target(
    model=model,
    head=head,
    tail=tail,
    triples_factory=triples_factory)

    labels = df.df['relation_label'].tolist()[:k]
    scores = df.df['score'].tolist()[:k]
    
    return labels, scores

def predict_missing_tail(dataset:str, model, head, relationship, k=5):

    triples_factory = get_dataset(dataset=dataset).training

    df = predict.predict_target(
    model=model,
    head=head,
    relation=relationship,
    triples_factory=triples_factory)

    labels = df.df['tail_label'].tolist()[:k]
    scores = df.df['score'].tolist()[:k]
    
    return labels, scores

def predict_triplet_score(dataset:str, model, head, relationship,tail):

    triples_factory = get_dataset(dataset=dataset).training

    prediction = predict.predict_triples(
    model=model,
    triples_factory=triples_factory,
    triples=(head,relationship,tail),batch_size=1)

    return prediction.scores[0]


# similarity functions
def entity_cosine_similarity(dataset,model,entity1,entity2):

    embedding1=calculate_embeddings(dataset,model,entity1)
    embedding2=calculate_embeddings(dataset,model,entity2)

    #print(embedding1)
    #print(embedding2)

    similarity = F.cosine_similarity(embedding1, embedding2, dim=0)

    return similarity.item()

def multiple_entity_cosine_similarity(dataset,model,entity_list1,entity_list2,similarity_metric="average"):
    
    similarity_matrix = np.zeros((len(entity_list1), len(entity_list2)))

    for i in range(len(entity_list1)):
        for j in range(len(entity_list2)):
            similarity_matrix[i, j] = entity_cosine_similarity(dataset,model,entity_list1[i],entity_list2[j])
    
    match similarity_metric:
        case "average":
            similarity = np.mean(similarity_matrix)
        case "min":
            similarity = np.min(similarity_matrix)
        case "max":
            similarity = np.max(similarity_matrix)
        case "median":
            similarity = np.median(similarity_matrix)
        case "sum":
            similarity = np.sum(similarity_matrix)
    
    print(similarity_matrix)
    print(similarity)

    return similarity

def embeddings_cosine_similarity(embedding1,embedding2):

    embedding1=torch.tensor([float(i) for i in embedding1])
    embedding2=torch.tensor([float(i) for i in embedding2])

    return F.cosine_similarity(embedding1, embedding2, dim=0).item()

def multiple_embedding_cosine_similarity(embeddings1,embeddings2,similarity_metric="average"):
    
    embeddings1=[torch.tensor(t) for t in embeddings1]
    embeddings2=[torch.tensor(t) for t in embeddings2]

    similarity_matrix = np.zeros((len(embeddings1), len(embeddings2)))

    for i in range(len(embeddings1)):
        for j in range(len(embeddings2)):
            similarity_matrix[i, j] = embeddings_cosine_similarity(embeddings1[i],embeddings2[j])
    
    match similarity_metric:
        case "average":
            similarity = np.mean(similarity_matrix)
        case "min":
            similarity = np.min(similarity_matrix)
        case "max":
            similarity = np.max(similarity_matrix)
        case "median":
            similarity = np.median(similarity_matrix)
        case "sum":
            similarity = np.sum(similarity_matrix)

    return similarity


# overlapping and center operations: COMPARAR CON CORRECCIONES EN EL TFG

def calculate_geometric_median(embeddings):
    embeddings=np.array(embeddings)
    def objective_function(point):
        return np.sum(cdist([point], embeddings)) 
    
    result = minimize(objective_function, np.mean(embeddings, axis=0))

    #media geométrica: minimiza la distancia a todas las entidades y devuelve un punto que equidista de todas
    geometric_median = result.x

    # CÁLCULO DEL RADIO
    distances_to_geometric_median = np.linalg.norm(embeddings - geometric_median, axis=1)
    r_median = np.max(distances_to_geometric_median)

    return geometric_median, r_median

def calculate_centroid(embeddings):
    #media aritmética: coordenadas resultantes de hacer la media de todas las entidades
    centroid = np.mean(embeddings,axis=0)
    distances_to_centroid = np.linalg.norm(embeddings - centroid, axis=1)
    r_centroid = np.max(distances_to_centroid)
    return centroid,r_centroid

def calculate_centers(embeddings, center_type="all"):
    '''If center_type == all returns all centers.
    implemented centers: centroid, geometric'''

    embeddings=np.array(embeddings) # transformar la lista de floats de fastapi a array de numpy

    centers = {}

    if center_type in {"geometric_median", "all"}:
        geometric_median, r_median = calculate_geometric_median(embeddings=embeddings)
        centers["geometric_median"] = {"point": torch.tensor(geometric_median), "radio": r_median}

    if center_type in {"centroid", "all"}:
        centroid, r_centroid = calculate_centroid(embeddings=embeddings)
        centers["centroid"] = {"point": torch.tensor(centroid), "radio": r_centroid}

    if center_type in centers:
        return {center_type: centers[center_type]}
    elif center_type == "all":
        return centers
    else:
        raise ValueError(
            f"Unrecognized center_type: '{center_type}'.\nAllowed values are: 'all', 'centroid', 'geometric_median'."
        )
    
def calculate_overlap(center1,center2,radius1,radius2):
    #distance between the centers of the circles
    distance = np.linalg.norm(np.array(center2) - np.array(center1))

    if distance >= radius1 + radius2:
        return 0

    elif distance <= abs(radius1 - radius2):
        return min(np.pi * radius1**2, np.pi * radius2**2) / max(np.pi * radius1**2, np.pi * radius2**2)

    else:
        overlap = (radius1**2 * np.arccos((distance**2 + radius1**2 - radius2**2) / (2 * distance * radius1))
                   + radius2**2 * np.arccos((distance**2 + radius2**2 - radius1**2) / (2 * distance * radius2))
                   - 0.5 * np.sqrt((-distance + radius1 + radius2) * (distance + radius1 - radius2) * (distance - radius1 + radius2) * (distance + radius1 + radius2)))
        
        return overlap / (np.pi * min(radius1**2, radius2**2))

if __name__ == "__main__": 
    #train_embedding_model("wikidata5m", embedding_dim=32,embedding_model="Transr",num_epochs=5,device="gpu",plot_results=True)
    pass