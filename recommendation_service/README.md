# 📰 Recommendation Service

The Recommendation Service is a simple downstream application of the Knowledge Graph Embedding API and serves as a content recommendation use-case. It estimates the relevance between two news articles, one currently visited and a candidate article, by analyzing the entities mentioned in each and comparing their semantic similarity using their Knowledge Graph Embeddings. To do so, it leverages the KGEM-API to compute features based on the embedded representation of these entities.

Unlike other approaches that use document-level embeddings, this service directly operates on the entities detected in the articles. Using their embeddings from a knowledge graph, the API calculates multiple inter-entity features, which are then processed by a ML model to predict a relevance score.


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
  "news1_entities": ["Q6279", "Q1223"], # Joe Biden, Washington
  "news2_entities": ["Q22686", "Q35525"], # Donald Trump, White House
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
    "cluster 1": ["Q6279", "Q1223"], # Joe Biden, Washington
    "cluster 2": ["Q22686", "Q35525"], # Donald Trump, White House
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
    "cluster 1": ["Q6279", "Q1223"], # Joe Biden, Washington
    "cluster 2": ["Q22686", "Q35525"], # Donald Trump, White House
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
- **Regression**: Returns a float score (e.g., `0.79`) indicating relevance.

## Used Features

The features are automatically selected from the trained ML instance. We obtain them by accessing the Knowledge Graph Embedding API endpoints.

| Feature Name         | Description |
|----------------------|-------------|
| `mean`               | Average cosine similarity between all entity pairs. |
| `max`                | Maximum cosine similarity between all entity pairs. |
| `median`             | Median cosine similarity between all entity pairs. |
| `min`                | Minimum cosine similarity between all entity pairs. |
| `sum`                | Sum of cosine similarities between all entity pairs. |
| `centroid-Rmean`     | Overlap ratio between entity clusters using centroid as center and mean radius. |
| `centroid-Rmedian`   | Overlap ratio using centroid and median radius. |
| `centroid-Rmax`      | Overlap ratio using centroid and max radius. |
| `geometric`          | Overlap ratio using geometric median and mean radius. |
| `distance_centroids` | Euclidean distance between centroids of both entity clusters. |
| `distance_geometric` | Euclidean distance between geometric medians of both entity clusters. |

## Model Training

> ℹ️ **info**  
> Check the [`model_training.ipynb`](./model_training.ipynb) notebook for further understanding on how the ML models are trained.

The data we are using for training consist on:

1. **Article Pairs Dataset**: A collection of article pairs, each annotated with a relevance label. The label can be:
   - **Binary**: Indicating whether one article is a good recommendation for the other (e.g., `0` for not relevant, `1` for relevant).
   - **Continuous**: Representing a similarity score between the articles (e.g., a value between `0.0` and `1.0`).

   **Example Structure**:
   ```plaintext
   | art1 | art2 | continuous_score | binary_score |
   |------|------|------------------|--------------|
   | 101  | 205  | 0.85             | 1            |
   ```

2. **News Metadata Dataset**: Contains metadata for each article. For this use case, we specifically need the entities identified within the article. Each entity should be linked to a unique graph identifier.

   **Example Structure**:
   ```plaintext
   | article_id | entities |
   |------------|----------|
   | 101        | [{'wikidataId': 'Q312'}, {'wikidataId': 'Q2796'}] |
   ```

While this system is compatible with any dataset following the above structure, for our implementation, we utilized the **CNRec dataset**. CNRec provides 2,700 pairs of news articles, each annotated by human evaluators for similarity and recommendation suitability. The dataset includes:

For more details on CNRec, refer to the original publication:

> Kevin Joseph and Hui Jiang. "Content based News Recommendation via Shortest Entity Distance over Knowledge Graphs." *Proceedings of the 28th ACM International Conference on Information and Knowledge Management (CIKM '19)*, 2019. [https://doi.org/10.1145/3308560.3317703](https://doi.org/10.1145/3308560.3317703)

The CNRec dataset is publicly available at: [https://github.com/kevinj22/CNRec](https://github.com/kevinj22/CNRec)
