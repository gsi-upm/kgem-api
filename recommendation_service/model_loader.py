from fastapi import HTTPException
import joblib

class ModelRegistry:
    def __init__(self):
        
        self._models = {"classification":{}, # classification stores models for binary classification (0 - Bad Recommendation, 1 - Good Recommendation)
                        "regression":{}} # regression stores models for 
    
    def load_model_metadata(self):
        # carga los metadatos del json sobre el modelo para la caja de cristal (quién ha entrenado el modelo, dataset, fecha, etc)
        raise NotImplementedError
    
    def load_model(self, model_name: str, mode: str="classification"):
        if model_name not in self._models[mode]:
            try:
                model_path = f"models/{mode}/{model_name}.joblib"
                self._models[mode][model_name] = joblib.load(model_path)
            except FileNotFoundError as e:
                raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.\nDetails:{e}")
        return self._models[mode][model_name]

model_registry = ModelRegistry()

def get_model(model_name,mode="classification"):
    return model_registry.load_model(model_name,mode)