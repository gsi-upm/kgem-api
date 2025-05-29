# kgem-api
An API for easy access to Knowledge Graphs and operations with Knowledge Graph Embedding Models. This API provides a set of endpoints for interacting with and performing operations and requests on different Knowledge Graphs. It supports operations for listing available graph datasets, retrieving embedding from a certain entity, link prediction and more.

## API Documentation

### Graphs

| Endpoint | Method | Description |
|---------|--------|-------------|
| `/` | GET | Returns a list of available graph datasets. |
| `/{graph_name}` | GET | Returns metadata for the specified graph. |
| `/{graph_name}/entity_list` | GET | Returns a list of all entities in the graph. |
| `/{graph_name}/relationship_list` | GET | Returns a list of all relationships in the graph. |
| `/refresh` | POST | Refreshes and updates the list of available graphs. |

---

### Embeddings

| Endpoint | Method | Parameters | Description |
|----------|--------|------------|-------------|
| `/` | GET | — | Lists all available embedding models. |
| `/by-entity/{graph_name}/{embedding_model}/{entity}` | GET | `graph_name`, `embedding_model`, `entity` | Returns the embedding representation of the specified entity within a specific graph and embedding model. |
| `/closest-entity/{graph_name}/{embedding_model}` | POST | `graph_name`, `embedding_model` <br> Body: `embedding: list[float]`, `k: int = 5` | Returns the top-k closest entities to a given embedding vector. |

---


### Predictions

| Endpoint | Method | Parameters | Description |
|----------|--------|------------|-------------|
| `/link/{graph_name}/{embedding_model}/{head}/{tail}` | GET | `graph_name`, `embedding_model`, `head`, `tail`, `k: int = 5` | Predicts the k most likely relationships between two entities. |
| `/entity/{graph_name}/{embedding_model}/{head}/{relationship}` | GET | `graph_name`, `embedding_model`, `head`, `relationship`, `k: int = 5` | Predicts the k most likely tail entities for a given head and relation. |
| `/probability/{graph_name}/{embedding_model}/{head}/{relationship}/{tail}` | GET | `graph_name`, `embedding_model`, `head`, `relationship`, `tail` | Returns a score indicating the plausibility of a triplet. |

---


### Similarities

| Endpoint | Method | Parameters | Description |
|----------|--------|------------|-------------|
| `/closest-entities/{graph_name}/{embedding_model}/{entity}` | GET | `graph_name`, `embedding_model`, `entity`, `k: int = 5` | Uses cosine similarity to return top-k closest/most similar entities to the specified entity. |
| `/cosine-similarity/entities/{graph_name}/{embedding_model}/{entity1}/{entity2}` | GET | `graph_name`, `embedding_model`, `entity1`, `entity2` | Computes cosine similarity between two entities. |
| `/cosine-similarity/embeddings` | POST | Body: `embedding_1: list[float]`, `embedding_2: list[float]` | Computes cosine similarity between two embedding vectors. |
| `/cosine-similarity/entities/multiple/{graph_name}/{embedding_model}/{metric}` | POST | `graph_name`, `embedding_model`, `metric` <br> Body: `entity_list1: list[str]`, `entity_list2: list[str]` | Computes and aggregates cosine similarities between two entity groups. |
| `/cosine-similarity/embeddings/multiple/{metric}` | POST | `metric` <br> Body: `embedding_list1: list[float]`, `embedding_list2: list[float]` | Computes and aggregates cosine similarities between two groups of embeddings. |

---

### Centers

| Endpoint | Method | Parameters | Description |
|----------|--------|------------|-------------|
| `/{graph_name}/{embedding_model}` | POST | `graph_name`, `embedding_model`, `raise_on_missing` <br> Body: `entity_list1: list[str]` | Computes both centroid and geometric median of the entities. Returns embeddings, radii, and closest entity to each center. |
| `/overlap/{graph_name}/{embedding_model}/{center_type}/{radius_type}` | POST | `graph_name`, `embedding_model`, `center_type`, `radius_type`, `raise_on_missing` <br> Body: `entity_list1: list[str]`, `entity_list2: list[str]` | Computes the overlap ratio between two entity clusters. |
| `/distance/{graph_name}/{embedding_model}/{center_type}` | POST | `graph_name`, `embedding_model`, `center_type`, `raise_on_missing` <br> Body: `entity_list1: list[str]`, `entity_list2: list[str]` | Computes the Euclidean distance between the centers of two clusters. |

---

## 📌 Notes

- **Graph Names & Embedding Models**: Use values returned from `/graphs` and `/embedding_models`.
- **Center Types**: `centroid`, `geometric_median`
- **Similarity Agg. Metric Options**: `mean`, `median`, `max`, `min`, `sum`
- **Radius Options**: `mean` (mean distance from the center to all entities), `median` (median distance from the center to all entities from the cluster), `max`(distance to the furthest entity form the center)
- **raise_on_missing**: If `True`, raises an error if there are any missing entities in the graph. If `False`, it computes the similarity aggregations, centers or overlapping ratios omitting the missing entities.

---

## Training your Knowledge Graph Embedding Models
To use the API, trained Knowledge Graph Embedding models must be available. These models are stored in the `models/` directory by default. If you encounter an error like `404: Model <model_name> not found.`, it indicates that the model is either missing from this directory or not yet trained. 

The `model_training.py` script provides an easy way to train and save embedding models using [PyKEEN](https://pykeen.readthedocs.io/en/stable/index.html). This library supports a wide range of [Knowledge Graph Embedding Models](https://pykeen.readthedocs.io/en/stable/reference/models.html) and [Knowledge Graph Datasets](https://pykeen.readthedocs.io/en/stable/reference/datasets.html)

### Step by step training example
#### 1. **Choose Your Knowledge Graph Dataset**
- You can use a built-in dataset from PyKEEN (e.g., nations, fb15k) or provide your own dataset in TSV format.
- If you use a custom set of triples, it must be a TSV file that follows this structure:
```
subject1   predicate1   object1
subject2   predicate2   object2
```
#### 2. **Configure the Training Parameters**
Open the `model_training.py` script and customize the training parameters as you wish. Some of the key options include:

**Dataset Source**:
- Use a predefined PyKEEN dataset by setting `triples_from_dataset=True` and specifying the dataset name (e.g., `dataset="nations"`).
- Load a custom dataset from a file by setting `triples_from_path=True` and providing the path (`triples_path="path/to/triples.tsv`).

**Model Settings**:
- Specify the embedding model (e.g., `TransE`, `ComplEx`) via `embedding_model`.
- Adjust hyperparameters like embedding dimensions, loss function, learning rate, and number of epochs.

**Saving Options**:
- Models are saved to the `models/` directory by default. You can change this with `models_route`.

Example Configuration for the Nations Dataset:
```python
train_embedding_model(
  graph_name="nations",
  dataset="nations",
  embedding_model="TransE",
  embedding_dim=50,
  num_epochs=100,
  models_route="models",
  triples_from_dataset=True
)
```
Refer to the [Pykeen Pipeline Documentation](https://pykeen.readthedocs.io/en/stable/api/pykeen.pipeline.pipeline.html#pykeen.pipeline.pipeline) for more details.

#### 3. **Train the model**
Execute the script to start the training process:

```bash
python scripts/model_training.py
```
This will output training progress and save the trained model in the `models/` directory (e.g., `models/nations/`).

#### 4. **Validate and use the model**
The trained model is loaded dynamically in the API using the `kge_model_loader.py` script. The `get_model` function provides a convenient way to load models by combining the graph name and embedding model:
```python
from kge_model_loader import get_model

model = get_model(graph_name="nations", embedding_model="TransE")
```
### Use pretrained embeddings

Sometimes you may prefer to use good-quality pretrained embeddings from external sources instead of training your own from scratch.

For this purpose, you can use the script `load_pretrained.py`, which takes a `.json` file like:

```json
{
  "Q42": [0.1, 0.2, 0.3],
  "Q123": [0.4, 0.5, 0.6]
}
```
Here the keys refer to the entity ids in the graph and the values represent its embedding vector.

Edit the script to set:

```python
path_to_pretrained_embeddings = "pretrained_embeddings/your_embeddings.json"
embedding_model = "transe" # this is just to save the model used for the pretrained embeddings
output_dir = f"models/wikidataAMORset_{embedding_model}"
```

Then run:

```bash
python scripts/load_pretrained.py
```

## Try it out!

The `demo.ipynb` Jupyter Notebook provides a short example on how you can access the API and create your own programs with Python. This notebook includes examples of how to interact with the API, make requests, and handle responses. It is a great starting point for understanding how to use the API in your own Python projects.
