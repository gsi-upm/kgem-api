import torch
from pykeen.models import TransE
from pykeen.datasets import get_dataset
from pykeen.nn.init import PretrainedInitializer

import pickle


# CARGAR MODELO DE OTRA LIBRERÍA
print("LOADING PRETRAINED MODEL")
with open("models/pretrained/transe_wikidata5m.pkl", "rb") as fin:
    model = pickle.load(fin)

pretrained_entity_embeddings = model.solver.entity_embeddings
pretrained_relationship_embeddings = model.solver.relation_embeddings


print("LOADING DATASET")
dataset = get_dataset(dataset="wikidata5m")

print("loaded dataset succesfully.")

print("LOADING PYKEEN MODEL")
model = TransE(
    triples_factory=dataset.training,
    embedding_dim=pretrained_entity_embeddings.shape[-1],  #match dimension of pre-trained embedding
    entity_initializer=PretrainedInitializer(pretrained_entity_embeddings),
    relation_initializer=PretrainedInitializer(pretrained_relationship_embeddings)
)

save_path="models/wikidata5m_transe"
model.save_to_directory(save_path)

print("SUCCESS")