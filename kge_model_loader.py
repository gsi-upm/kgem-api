import torch
from fastapi import HTTPException

class ModelRegistry:
    def __init__(self):
        self._models = {}
    
    def load_model(self, name: str):
        '''Lazy Knowledge Graph Embedding Model loader'''
        if name not in self._models:
            try:
                model_path = f"models/{name}/trained_model.pkl"
                self._models[name] = torch.load(model_path)
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
        return self._models[name]

model_registry = ModelRegistry()
embedding_model = "transe"

def get_model(graph_name:str, embedding_model:str):
    model_name=f"{graph_name}_{embedding_model}"
    return model_registry.load_model(model_name)
