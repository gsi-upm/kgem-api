import torch
from fastapi import HTTPException

class ModelRegistry:
    """
    A registry to manage Knowledge Graph Embedding Models.
    
    This class provides a lazy-loading mechanism for models, ensuring that models are only loaded
    into memory when requested. It caches the loaded models for efficient reuse, reducing 
    overhead during repeated API calls.
    """
    def __init__(self):
        self._models = {} # dictionary to store models in this format {model_name:model_instance}
    
    def load_model(self, name: str):
        """
        Load a Knowledge Graph Embedding model by name.
        
        Args:
            name (str): The name of the model to load.
            
        Returns:
            torch.nn.Module: The loaded PyTorch model.
            
        Raises:
            HTTPException: If the model file is not found in the expected directory.
        """
        if name not in self._models:
            try:
                model_path = f"models/{name}/trained_model.pkl"
                self._models[name] = torch.load(model_path)
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
        return self._models[name]

model_registry = ModelRegistry()
embedding_model = "transe" # change the embedding model used in the api

def get_model(graph_name:str, embedding_model:str):
    """
    Helper function to retrieve a Knowledge Graph Embedding model.
    
    This function combines the graph name and embedding model type to construct the model 
    identifier (e.g., "nations_transe") and uses the ModelRegistry for loading. It is 
    primarily used by API routers, centralizing model retrieval logic to improve 
    code organization and scalability.
    
    Args:
        graph_name (str): The name of the knowledge graph.
        embedding_model (str): The name of the embedding model (e.g., 'transe', 'complex').
        
    Returns:
        torch.nn.Module: The loaded model instance.
    """
    model_name=f"{graph_name}_{embedding_model}"
    return model_registry.load_model(model_name)
