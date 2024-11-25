from pydantic_models import Graph
from fastapi import APIRouter, HTTPException

router = APIRouter()



# defining a few graphs for testing the API
nations = Graph(id=0, name="Nations", description="Graph representation of relationships between countries.", n_entities=14, n_triples=1992, n_relations=55, 
                dataset_url="")

dbpedia50 = Graph(id=1, name="DBpedia50", description="Graph containing structured information extracted from Wikipedia articles.", n_entities=24624, n_triples=34421, n_relations=351, 
                dataset_url="https://raw.githubusercontent.com/ZhenfengLei/KGDatasets/master/DBpedia50")

wikidata = Graph(id=2, name="Wikidata", description="Graph containing structured information extracted from Wikipedia articles.", n_entities=4594149, n_triples=20624239, n_relations=822, 
                dataset_url="https://zenodo.org/record/5546383/files/wikidata5m_transductive.tar.gz")

graphs=[nations,dbpedia50,wikidata]




@router.get("/")
def list_graphs() -> list[Graph]:
    '''Retrieves a list of all available graph datasets detailing its ID, name, description, number of entities, triples, and relations..'''
    return graphs

@router.get("/{graph_id}")
def get_graph(graph_id: int) -> Graph:
    '''Returns details of a specific graph dataset identified by its ID. It includes ID, name, description, number of entities, triples, and relations.
    '''
    if graph_id < 0 or graph_id >= len(graphs):
        raise HTTPException(status_code=404, detail="Graph not found")
    return graphs[graph_id]


# @router.get("/graphs/by_entity/{entity}")
# def list_graphs_with_entity(entity: str) -> list[Graph]:
#     '''Returns a list of all the graphs that contain the requested entity.
#     NOT IMPLEMENTED'''

#     graphs=[]

#   return graphs