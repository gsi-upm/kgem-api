import os
from typing import List
from pykeen.datasets import get_dataset
from pydantic_models import Graph


MODELS_DIR = "models" 

def get_available_embedding_models() -> List[str]:
    """Returns a list of available embedding models."""
    if not os.path.isdir(MODELS_DIR):
        return []
    subdirs = os.listdir(MODELS_DIR)
    models = {subdir.split('_')[1] for subdir in subdirs if "_" in subdir}
    return sorted(models)

def get_available_graphs_from_models_folder() -> List[str]:
    """Scans the models/ directory and returns a list of graph names."""
    if not os.path.isdir(MODELS_DIR):
        return []
    subdirs = os.listdir(MODELS_DIR)
    graphs = {subdir.split('_')[0] for subdir in subdirs if "_" in subdir}
    return sorted(graphs)


DATASET_INFO = {
    'nations': {
        'description': 'Graph representation of relationships between countries.',
        'dataset_url': 'https://github.com/ZhenfengLei/KGDatasets/tree/master/Nations',
    },
    'dbpedia50': {
        'description': 'Graph containing structured information extracted from Wikipedia articles.',
        'dataset_url': 'https://github.com/ZhenfengLei/KGDatasets/tree/master/DBpedia50',
    },
    'wikidata5m': {
        'description': 'A million-scale knowledge graph dataset integrating Wikidata and Wikipedia pages.',
        'dataset_url': 'https://deepgraphlearning.github.io/project/wikidata5m',
    },
    'wd50kt': {
        'description': 'The triples-only version of WD50K, containing structured information extracted from Wikidata.',
        'dataset_url': 'https://github.com/migalkin/WD50K',
    },
    'codexlarge': {
        'description': 'The CoDEx large dataset from Safavi and Koutra (2020).',
        'dataset_url': 'https://github.com/tsafavi/codex',
    },
    'codexsmall': {
        'description': 'The CoDEx small dataset from Safavi and Koutra (2020).',
        'dataset_url': 'https://github.com/tsafavi/codex',
    },
    'countries': {
        'description': 'The Countries dataset.',
        'dataset_url': 'https://github.com/ZhenfengLei/KGDatasets/tree/master/Countries',
    },
    'wikidataAMORset':{
        'description': 'A dummy subset from wikidata5m for using pre-trained embeddings of the entities in amor-graph.',
        'dataset_url': ''
    }
}

def get_graph_metadata(graph_name: str):
    try:     
        if graph_name.lower() == "wikidataamorset": # Special case: custom graph not available in pykeen
            metadata = {
                "name": graph_name,
                "n_entities": 325,
                "n_relations": 1,
                "n_triples": 0,
            }
        else:
            dataset = get_dataset(dataset=graph_name) # Assume most of the graphs will be available in pykeen
            metadata = {
                "name": dataset.metadata["name"],
                "n_entities": dataset.num_entities,
                "n_relations": dataset.num_relations,
                "n_triples": sum(factory.num_triples for factory in [dataset.training, dataset.testing, dataset.validation] if factory),
            }
        
        # add description and dataset_url from DATASET_INFO if available
        if graph_name in DATASET_INFO:
            metadata["description"] = DATASET_INFO[graph_name]["description"]
            metadata["dataset_url"] = DATASET_INFO[graph_name]["dataset_url"]
        else:
            metadata["description"] = "No description available"
            metadata["dataset_url"] = "No URL available"
        
        print(f"\n returning metadata:{metadata}")
        return metadata
        
    except Exception as e:
        print(f"Error loading dataset {graph_name}: {e}")
        #raise e
    
graph_list = None  # Cached list of Pydantic Graph objects

def initialize_graph_list():
    """Initializes the global graph list by generating Pydantic Graph objects."""
    global graph_list
    graph_names = get_available_graphs_from_models_folder()
    graph_list = [Graph(**get_graph_metadata(graph)) for graph in graph_names]

def get_pydantic_graphs():
    """Retrieves the cached list of Pydantic Graph objects. Initializes it if not already done."""
    global graph_list
    if graph_list is None:
        initialize_graph_list()
    return graph_list

def refresh_graph_list():
    """Refreshes the global graph list by regenerating it."""
    initialize_graph_list()
    