# kgem-api
An API for easy access to Knowledge Graphs and operations with Knowledge Graph Embedding Models. This API provides a set of endpoints for interacting with and performing operations and requests on different Knowledge Graphs. It supports operations for listing available graph datasets, retrieving embedding from a certain entity, link prediction and more.

## API Documentation


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

Edit the script to set:

```python
path_to_pretrained_embeddings = "pretrained_embeddings/your_embeddings.json"
embedding_model = "transe"
output_dir = f"models/wikidataAMORset_{embedding_model}"
```

Then run:

```bash
python scripts/load_pretrained.py
```

The script will:
- Initialize a PyKEEN model with the pretrained vectors
- Save it to the specified `models/` folder (used later by the API)
- Require no actual triples or training (epochs=0)


## Try it out!

The `demo.ipynb` Jupyter Notebook provides a short example on how you can access the API and create your own programs with Python. This notebook includes examples of how to interact with the API, make requests, and handle responses. It is a great starting point for understanding how to use the API in your own Python projects.