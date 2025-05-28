# 📰 Recommendation Service

The Recommendation Service is a simple downstream application of the Knowledge Graph Embedding API and serves as a recommendation use-case for content relevance. It estimates the relevance between two news articles, one currently visited and a candidate article, by analyzing the entities mentioned in each and comparing their semantic similarity using their Knowledge Graph Embeddings. To do so, it leverages the KGEM-API to compute features based on the embedded representation of these entities.

Unlike other approaches that use document-level embeddings, this service directly operates on the entities detected in the articles. Using their embeddings from a knowledge graph, the API calculates multiple inter-entity features, which are then processed by a ML model to predict a relevance score.


## Used Features

The features are automatically selected from the trained ML instance. We obtain them by accessing the Knowledge Graph Embedding API endpoints.

| Feature Name         | Description |
|----------------------|-------------|
| `mean`               | Average cosine similarity between all entity pairs. |
| `max`                | Maximum cosine similarity between any entity pair. |
| `median`             | Median cosine similarity between all entity pairs. |
| `min`                | Minimum cosine similarity between all entity pairs. |
| `sum`                | Sum of cosine similarities between all entity pairs. |
| `centroid-Rmean`     | Overlap ratio between entity clusters using centroid as center and mean radius. |
| `centroid-Rmedian`   | Overlap ratio using centroid and median radius. |
| `centroid-Rmax`      | Overlap ratio using centroid and max radius. |
| `geometric`          | Overlap ratio using geometric median and mean radius. |
| `distance_centroids` | Euclidean distance between centroids of both entity clusters. |
| `distance_geometric` | Euclidean distance between geometric medians of both entity clusters. |


## Input Parameters for `POST /recommend`

| Parameter         | Type       | Description |
|------------------|------------|-------------|
| `news1_entities` | list[str]  | List of entities from the first article (e.g., `"Q23", "Q432"`). |
| `news2_entities` | list[str]  | List of entities from the second article. |
| `mode`           | enum       | `"classification"` or `"regression"` — determines the output type. |
| `model_name`     | enum       | Predictive model to use (e.g., `"random_forest"`). |
| `graph`          | str        | Name of the knowledge graph dataset (e.g., `"wikidata5m"`). |
| `embedding_model`| str        | Name of the embedding model (e.g., `"transe"`). |

---


## Example Request Body

```json
{
  "news1_entities": ["Joe Biden", "Washington"],
  "news2_entities": ["Donald Trump", "White House"],
  "mode": "classification",
  "model_name": "random_forest",
  "graph": "wikidataAMORset",
  "embedding_model": "transe"
}
```

## Example Responses

### Classification Mode

```json
{
  "relevance_score": 1,
  "model_used": "random_forest",
  "mode": "classification",
  "graph": "wikidataAMORset",
  "embedding_model": "transe",
  "used_entities": {
    "cluster 1": ["Joe Biden", "Washington"],
    "cluster 2": ["Donald Trump", "White House"]
  },
  "features": {
    "mean": 0.46,
    "centroid-Rmean": 0.18,
    "distance_centroids": 1.05
  }
}
```

### Regression Mode

```json
{
  "relevance_score": 0.79,
  "model_used": "random_forest",
  "mode": "regression",
  "graph": "wikidataAMORset",
  "embedding_model": "transe",
  "used_entities": {
    "cluster 1": ["Joe Biden", "Washington"],
    "cluster 2": ["Donald Trump", "White House"]
  },
  "features": {
    "mean": 0.46,
    "centroid-Rmean": 0.18,
    "distance_centroids": 1.05
  }
}
```
### Score Output Types

- **Classification**: Returns `0` (irrelevant) or `1` (relevant).
- **Regression**: Returns a float score (e.g., `0.79`) indicating nuanced relevance.

training
- notebook de entrenamiento
- citar tfg para saber cómo se han entrenado lso modelos de ML