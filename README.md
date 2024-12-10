# kgem-api-gsi
An API for easy access to Knowledge Graphs and operations with Knowledge Graph Embedding Models. This API provides a set of endpoints for interacting with and performing operations and requests on different Knowledge Graphs. It supports operations for listing available graph datasets, retrieving embedding from a certain entity, link prediction and more.

## Quick set up
Clone this repository:
```bash
$ cd git clone https://github.com/rjvillen/kgem-api-gsi.git
```
Create a virtual environment in the project directory. The example uses venv. However, you can use any virtual environment of your preference. 
```bash
$ cd /path/to/project
$ python3 -m venv venv && source venv/bin/activate
$ pip install requirements.txt
```
Use this command to run the API:
```bash
$ uvicorn  main:app --reload
```
The command `uvicorn main:app --reload` starts a development server for the FastAPI application. It tells Uvicorn to look for the app instance in the `main.py` file and serves it on the default port (8000). The `--reload` flag enables auto-reloading, so the server restarts automatically whenever you make changes to the code. Check [FastAPI documentation](https://fastapi.tiangolo.com/) for further information.

The API should now be accessible at [http://127.0.0.1:8000](http://127.0.0.1:8000).

## API Interactive Documentation
FastAPI provides an interactive documentation available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

After clicking on the link you should see something like this:
![image](https://github.com/user-attachments/assets/88abb0e0-b170-49b8-ae69-2aa52126d74d)

This interface allows you to explore all available endpoints, see what data each one accepts, and view the expected responses. You can even test the API directly from your browser by sending requests and seeing the results in real-time, in a user-friendly way :)

Visit [fastAPI Interactive Docs Page](https://fastapi.tiangolo.com/#interactive-api-docs) for further details on this topic.

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
[!WARNING]
>The parameter `embedding_model` is **hardcoded** in this version of the code. 
>If you need to modify this variable, you must manually update it in the `kge_model_loader.py` file.

## Try it out!
- demo scripts
- importante falta información para la gente que no sabe las dimensiones de los embeddigs, cómo acceden a los grafos? poner ejemplos, cómo saben el nombre de las entidades? ilustrar con ejemplo
