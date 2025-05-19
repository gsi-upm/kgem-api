from fastapi import APIRouter
from kge_model_loader import get_model
from EmbeddingsLib import calculate_entity_from_embedding, calculate_centers, calculate_embeddings, calculate_overlap, compute_center_distance

router = APIRouter()



@router.post("/{graph_name}/{embedding_model}")
def get_centers(graph_name:str,embedding_model:str,entity_list1:list[str]):
    '''Computes the center point in the embedding space for a given list of entities. 
    The response includes two types of center points: the geometric median and the centroid. 
    Both center points are represented by their embedding vectors and additional metadata such 
    as the cluster radius (can be meand, median or maximum distance to the entity emebddings) 
    and the closest entity to that point.'''

    model=get_model(graph_name,embedding_model)

    embeddings=[calculate_embeddings(model,e) for e in entity_list1]
    centers = calculate_centers(embeddings)

    for center_type in centers:

        # add closest entity
        centers[center_type]["closest entity"]=calculate_entity_from_embedding(model,centers[center_type]["point"],k=1)[0][0] # no devolver las similitudes, solo la primera entidad de la lista de entidades

        #transformar en numeros decimales para poder usarlos en la API
        centers[center_type]["point"]=[float(i) for i in centers[center_type]["point"]]
        centers[center_type]["radii"]={k:float(v) for k,v in centers[center_type]["radii"].items()}

    return {"input entities":entity_list1} | centers


@router.post("/overlap/{graph_name}/{embedding_model}/{center_type}/{radius_type}")
def get_overlap(graph_name:str,embedding_model:str,entity_list1:list[str],entity_list2:list[str],center_type:str,radius_type:str):
    '''Computes the overlapping area between two groups of entities based on their centers and radii. 
    The center can be specified as either "centroid" (center of mass) or "geometric median" (a more central point). 
    The radius can be specified as either "mean", "median" or "max" distances between both centers.
    The choice of center and radius affects the size and shape of the overlapping area.'''
    model=get_model(graph_name,embedding_model)

    embeddings1=[calculate_embeddings(model,e) for e in entity_list1]
    centers_1 = calculate_centers(embeddings1,center_type)

    embeddings2=[calculate_embeddings(model,e) for e in entity_list2]
    centers_2= calculate_centers(embeddings2,center_type)

    center_1=centers_1[center_type]["point"]
    center_2=centers_2[center_type]["point"]

    r_1=centers_1[center_type]["radii"][radius_type]
    r_2=centers_2[center_type]["radii"][radius_type]

    overlap=calculate_overlap(center_1,center_2,r_1,r_2)

    #transformar en una lista de numeros decimales para poder usarlos en la API
    center_1=[float(i) for i in center_1]
    center_2=[float(i) for i in center_2]

    r_1=float(r_1)
    r_2=float(r_2)
    
    overlap=float(overlap)

    return {"cluster 1":{"input entities":entity_list1,"center":center_1, "radius":r_1}, 
            "cluster 2":{"input entities":entity_list2,"center":center_2, "radius":r_2},
            "overlap-ratio":overlap}


# nuevo endpoint que devuelva al distancia entre los centros de ambos clusters
@router.post("/distance/{graph_name}/{embedding_model}/{center_type}")
def get_center_distance(graph_name:str,embedding_model:str,entity_list1:list[str],entity_list2:list[str],center_type:str):
    '''Computes the euclidean distance between the centers of two clusters of entities.'''
    
    model=get_model(graph_name,embedding_model)

    embeddings1=[calculate_embeddings(model,e) for e in entity_list1]
    centers_1 = calculate_centers(embeddings1,center_type)

    embeddings2=[calculate_embeddings(model,e) for e in entity_list2]
    centers_2= calculate_centers(embeddings2,center_type)

    center_1=centers_1[center_type]["point"]
    center_2=centers_2[center_type]["point"]
    
    distance = compute_center_distance(center_1,center_2)
    
    return {"cluster 1":{"input entities":entity_list1,"center":center_1}, 
            "cluster 2":{"input entities":entity_list2,"center":center_2},
            "euclidean distance":distance}
