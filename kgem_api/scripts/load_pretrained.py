''' Script to load pretrained embeddings into a PyKEEN model.

Input format is a JSON file with entity IDs as keys and their embeddings as values.
Example: {"Q42": [0.1, 0.2, ...], "Q123": [0.3, 0.4, ...]}'''

import json
import torch
from pykeen.triples import TriplesFactory
import numpy as np
from pykeen.pipeline import pipeline
from pykeen.nn.init import PretrainedInitializer

from torch.serialization import add_safe_globals
from pathlib import PosixPath

add_safe_globals([PosixPath])


# PARAMETERS
path_to_pretrained_embeddings = "pretrained_embeddings/wikidata5m_AMOR_embeddings.json"
embedding_model = "transe"  # embedding model for the pretrained embeddings
output_dir = f"models/wikidataAMORset_{embedding_model}"



# LOAD PRETRAINED EMBEDDINGS FOR ENTITIES
with open(path_to_pretrained_embeddings) as f:
    raw = json.load(f)
emb_dict = {qid: torch.tensor(vec, dtype=torch.float32) for qid, vec in raw.items()}
if not emb_dict:
    raise RuntimeError("El diccionario de embeddings está vacío.")

# mapping entity -> index
qids = list(emb_dict)

entity_to_id = {qid: idx for idx, qid in enumerate(qids)}
relation_to_id = {"__DUMMY_REL__": 0}  # relationship mapping should not be empty.

empty_triples = np.empty((len(emb_dict), 3), dtype=np.int32) # dummy triples, shape (0, 3)

tf = TriplesFactory(
    mapped_triples=empty_triples,
    entity_to_id=entity_to_id,      
    relation_to_id=relation_to_id,  
)

# weight tensor (N × d)
N = len(qids)
d = next(iter(emb_dict.values())).numel()
weights = torch.stack([emb_dict[qid] for qid in qids], dim=0)  # shape (N, d)

# run pipeline with PretrainedInitializer (no training: epochs=0)
result = pipeline(
    training=tf,
    testing=tf,                        
    model=embedding_model,
    model_kwargs=dict(
        embedding_dim=d,
        entity_initializer=PretrainedInitializer(tensor=weights),
    ),
    training_kwargs=dict(num_epochs=0, batch_size=1),            
    random_seed=42,                                
)

# save
result.save_to_directory(output_dir)

# check entities and their embeddings
model = result.model
print("model state dict",model.state_dict().keys())   # use this to find the name of the weights