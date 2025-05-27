'''
Script to check which triple factories contain most of the entities from the graph.
'''

from pykeen.datasets import get_dataset # ,dataset_similarity
import warnings
import json

warnings.simplefilter('ignore', category=FutureWarning)

def load_dataset_entities(dataset_name):
    """
    Load entities from a PyKEEN dataset.
    """
    try:
        dataset = get_dataset(dataset=dataset_name)
        return set(dataset.entity_to_id.keys())
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        return set()

def calculate_coverage(entities, datasets):
    coverage_results = {}
    for dataset in datasets:
        dataset_entities = load_dataset_entities(dataset)
        print(f"Getting coverage for {dataset}.")
        if not dataset_entities:
            coverage_results[dataset] = 0
            continue
        
        covered_entities = set(entities) & dataset_entities
        coverage = len(covered_entities) / len(entities) * 100
        print(f"Coverage {coverage}")
        coverage_results[dataset] = coverage
    
    return coverage_results

def get_all_entities():
    """
    Read entities from file.
    """
    try:
        with open("entities_list.json", "r") as file:
            content = file.read().strip()
            entities = json.loads(content)
            if isinstance(entities, list):
                return list(set(entities))
            else:
                print("Error: The file does not contain a valid list.")
                return []
    except FileNotFoundError:
        print("Error: 'entities_list.txt' not found.")
        return []
    except json.JSONDecodeError:
        print("Error: Failed to parse the file. Ensure it contains a valid JSON list.")
        return []





if __name__=="__main__":

    entities = get_all_entities()
    # datasets = ['aristov4', 'biokg', 'ckg', 'cn3l', 'codexlarge', 'codexmedium', 'codexsmall', 'conceptnet', 'countries', 'cskg', 'db100k', 'dbpedia50', 'drkg', 'fb15k', 'fb15k237', 'globi', 'hetionet', 'kinships', 'nations', 'nationsliteral', 'ogbbiokg', 'ogbwikikg2', 'openbiolink', 'openbiolinklq', 'openea', 'pharmebinet', 'pharmkg', 'pharmkg8k', 'primekg', 'umls', 'wd50kt', 'wikidata5m', 'wk3l120k', 'wk3l15k', 'wn18', 'wn18rr', 'yago310'] # todos los datasets
    datasets = ["CodexSmall", "CodexMedium", "CodexLarge", "wd50kt", "Wikidata5M", "wk3l120k", 'wk3l120k', 'wk3l15k', 'wn18', 'wn18rr', 'yago310']

    coverage_results = calculate_coverage(entities, datasets)

    for dataset, coverage in coverage_results.items():
        print(f"Dataset: {dataset}, Coverage: {coverage:.2f}%")

    output_file = "coverage_results.json"
    with open(output_file, "w") as f:
        json.dump(coverage_results, f, indent=4)
    print(f"Coverage results saved to {output_file}")

