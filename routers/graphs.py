from pydantic_models import Graph
from fastapi import APIRouter, HTTPException
from pykeen.datasets import get_dataset
from graph_loading_utils import *

from kge_model_loader import get_model

router = APIRouter()



# defining a few graphs for testing the API
# nations = Graph(id=0, name="Nations", description="Graph representation of relationships between countries.", n_entities=14, n_triples=1992, n_relations=55, 
#                 dataset_url="")

# dbpedia50 = Graph(id=1, name="DBpedia50", description="Graph containing structured information extracted from Wikipedia articles.", n_entities=24624, n_triples=34421, n_relations=351, 
#                 dataset_url="https://raw.githubusercontent.com/ZhenfengLei/KGDatasets/master/DBpedia50")

# wikidata5m = Graph(id=2, name="Wikidata5m", description="Graph containing structured information extracted from Wikipedia articles.", n_entities=4594149, n_triples=20624239, n_relations=822, 
#                 dataset_url="https://zenodo.org/record/5546383/files/wikidata5m_transductive.tar.gz")

# wikidata50kt = Graph(id=3, name="wd50kt", description="The triples-only version of WD50K. Graph containing structured information extracted from Wikipedia articles.", n_entities=40107, n_triples=232344, n_relations=473, 
#                 dataset_url="")

# graphs=[nations,dbpedia50,wikidata5m,wikidata50kt]




@router.get("/")
def list_graphs() -> list[Graph]:
    '''Retrieves a list of all available graph datasets detailing its ID, name, description, number of entities, triples, and relations..'''
    return get_pydantic_graphs()

@router.get("/{graph_name}")
def get_graph(graph_name: str) -> Graph:
    '''Returns details of a specific graph dataset identified by its ID. It includes ID, name, description, number of entities, triples, and relations.
    '''
    graphs = get_pydantic_graphs()
    graph = next((g for g in graphs if g.name == graph_name), None)

    if graph is None:
        raise HTTPException(status_code=404, detail="Graph not found")
    return graph


@router.get("/{graph_name}/entity_list")
def get_entity_list(graph_name: str):
    try:
        # temporal:
        if graph_name == "wikidataAMORset":
            triples_factory = get_model(graph_name,"transe").triples_factory
        else:
            triples_factory = get_dataset(dataset=graph_name).training
        entity_labels = triples_factory.entity_id_to_label 
        return entity_labels
    except KeyError:
        raise HTTPException(status_code=404, detail="Invaid graph name. Please check available graphs at /graphs")

@router.get("/{graph_name}/relationship_list")
def get_relation_list(graph_name: str):
    try:
        # temporal:
        if graph_name == "wikidataAMORset":
            triples_factory = get_model(graph_name,"transe").triples_factory
        else:
            triples_factory = get_dataset(dataset=graph_name).training
        entity_labels = triples_factory.relation_id_to_label 
        return entity_labels
    except KeyError:
        raise HTTPException(status_code=404, detail="Invaid graph name. Please check available graphs at /graphs")



@router.post("/refresh")
def refresh_graphs():
    """Refreshes the list of available graph datasets."""
    refresh_graph_list()
    return {"message": "Graph list refreshed successfully"}

# @router.get("/graphs/by_entity/{entity}")
# def list_graphs_with_entity(entity: str) -> list[Graph]:
#     '''Returns a list of all the graphs that contain the requested entity.
#     NOT IMPLEMENTED'''

#     graphs=[]

#   return graphs