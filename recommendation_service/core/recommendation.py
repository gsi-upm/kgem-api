from model_loader import get_model
import requests
import pandas as pd
from fastapi import HTTPException

import os
from dotenv import load_dotenv


load_dotenv()
KGE_API_BASE = os.getenv("KGE_API_BASE")

def get_features(entities1, entities2, feature_names, graph_name="wikidataAMORset",embedding_model="transe"):
        
    payload = {
        "entity_list1": entities1,
        "entity_list2": entities2
    }

    map_endpoint = {
        # cosine similarity feature endpoints
        "mean": f"{KGE_API_BASE}/similarities/cosine-similarity/entities/multiple/{graph_name}/{embedding_model}/average",
        "max": f"{KGE_API_BASE}/similarities/cosine-similarity/entities/multiple/{graph_name}/{embedding_model}/max",
        "median": f"{KGE_API_BASE}/similarities/cosine-similarity/entities/multiple/{graph_name}/{embedding_model}/median",
        "sum": f"{KGE_API_BASE}/similarities/cosine-similarity/entities/multiple/{graph_name}/{embedding_model}/sum",
        "min": f"{KGE_API_BASE}/similarities/cosine-similarity/entities/multiple/{graph_name}/{embedding_model}/min",
        # overlapping features endpoints
        "centroid-Rmean": f"{KGE_API_BASE}/centers/overlap/{graph_name}/{embedding_model}/centroid/mean",
        "centroid-Rmedian": f"{KGE_API_BASE}/centers/overlap/{graph_name}/{embedding_model}/centroid/median",
        "centroid-Rmax": f"{KGE_API_BASE}/centers/overlap/{graph_name}/{embedding_model}/centroid/max",
        "geometric": f"{KGE_API_BASE}/centers/overlap/{graph_name}/{embedding_model}/geometric_median/mean",
        "distance_centroids": f"{KGE_API_BASE}/centers/distance/{graph_name}/{embedding_model}/centroid",
        "distance_geometric": f"{KGE_API_BASE}/centers/distance/{graph_name}/{embedding_model}/geometric_median"
    }
    
    features = {}
    
    for feature in feature_names:
        if feature in map_endpoint:
            #print(f"Fetching feature {feature} from {map_endpoint[feature]}")
            try:
                response = requests.post(map_endpoint[feature], json=payload)
                response.raise_for_status()
            except requests.exceptions.HTTPError as http_err:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"External API error for feature '{feature}': {response.text}")
            
            if response.status_code == 200:
                #print(response.json())
                if "similarities" in map_endpoint[feature]:
                    features[feature] = response.json()["similarity"]
                elif "overlap" in map_endpoint[feature]:
                    features[feature] = response.json()["overlap-ratio"]
                elif "distance" in map_endpoint[feature]:
                    features[feature] = response.json()["euclidean distance"]
            else:
                raise HTTPException(f"Error fetching feature {feature}: {response.text}")
        else:
            raise HTTPException(f"Feature {feature} not found in the mapping.")
    
    return features

def predict(data):
    model = get_model(data.model_name.value, data.mode.value)
    features = get_features(data.news1_entities, data.news2_entities, feature_names=model.feature_names_in_)
    score = model.predict(pd.DataFrame([features]))[0]
    return {
        "relevance_score": score,
        "model_used": data.model_name.value,
        "mode": data.mode.value,
        "features": features
    }




