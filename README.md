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
The command uvicorn main:app --reload starts a development server for the FastAPI application. It tells Uvicorn to look for the app instance in the main.py file and serves it on the default port (8000). The --reload flag enables auto-reloading, so the server restarts automatically whenever you make changes to the code. Check [FastAPI documentation](https://fastapi.tiangolo.com/) for further information.

The API should now be accessible at [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Access the API interactive documentation
FastAPI provides an interactive documentation available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

After clicking on the link you should see something like this:
![image](https://github.com/user-attachments/assets/88abb0e0-b170-49b8-ae69-2aa52126d74d)

This interface allows you to explore all available endpoints, see what data each one accepts, and view the expected responses. You can even test the API directly from your browser by sending requests and seeing the results in real-time, in a user-friendly way :)

Visit [fastAPI Interactive Docs Page](https://fastapi.tiangolo.com/#interactive-api-docs) for further details on this topic.

## Traing your Knowledge Graph Embedding Models
You may encounter the 404: Model <model_name> not found error.
Before using a model you should have a trained instance. Models are stored in the models folder by default.
Further information on pykeen documentation, check all available graphs and all available embedding models.
Example: training an instance of the nations knowledge graph with transe embedding model

the triples path should be a tsv file in which (incluir un ejemplo de fichero de triplas)

## Try it out!
- demo scripts

- importante falta información para la gente que no sabe las dimensiones de los embeddigs, cómo acceden a los grafos? poner ejemplos, cómo saben el nombre de las entidades?
