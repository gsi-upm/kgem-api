import torch
from fastapi import HTTPException

from torch.serialization import add_safe_globals
from pathlib import PosixPath

from pykeen.triples import TriplesFactory


add_safe_globals([PosixPath])    
    

class ModelRegistry:
    """
    A registry to manage Knowledge Graph Embedding Models.
    
    This class provides a lazy-loading mechanism for models, ensuring that models are only loaded
    into memory when requested. It caches the loaded models for efficient reuse, reducing 
    overhead during repeated API calls.
    """
    def __init__(self):
        self._models = {} # dictionary to store models in this format {model_name:model_instance}
        
    def _load_triples_factory(self,model_name):
        '''loads triples_factory into specified model from the model registry.'''
        training_triples_factory = TriplesFactory.from_path_binary(
        path=f'models/{model_name}/training_triples')
        
        return training_triples_factory
    
    def load_model(self, model_name: str):
        """
        Load a Knowledge Graph Embedding model by name.
        
        Args:
            model_name (str): The name of the model to load.
            
        Returns:
            torch.nn.Module: The loaded PyTorch model.
            
        Raises:
            HTTPException: If the model file is not found in the expected directory.
        """
        if model_name not in self._models:
            try:
                model_path = f"models/{model_name}/trained_model.pkl"
                if torch.cuda.is_available():         
                    self._models[model_name] = torch.load(model_path, weights_only=False)
                else:
                    self._models[model_name] = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
                
                # asignar al modelo las triplas que se usaron para entrenarlo
                # pykeen Model class permite guardar la triples_factory en un atributo del propio modelo
                self._models[model_name].triples_factory = self._load_triples_factory(model_name)
            
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found. Try a different combination of graph and embedding model.")
        return self._models[model_name]

model_registry = ModelRegistry()

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