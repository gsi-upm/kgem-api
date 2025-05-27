# Inferir una relación dada dos entidades o una entidad dada una relación y una entidad

import requests

base_url="http://localhost:8000"
#graph_name="dbpedia50"
graph_name="codexlarge"


#question="Qué país es aliado de israel?"
#question=("israel","aliado")


def infer_tail(question):

    head,relationship=question

    #obtener en qué grafos aparece la entidad?
    #obtener nombre de la entidad en el grafo en concreto?

    endpoint=f"/triplets/entity/{graph_name}/{head}/{relationship}"
    response=requests.get(f"{base_url}{endpoint}")

    if response.status_code == 200:
        data = response.json()
        for i in range(len(data)):
            print(data[i]["tail"])
    else:
        print(f"Error: {response.status_code}")


#question="Cuál es la relación entre China y USA?"
#question=("china","usa")


def infer_link(question):

    head,tail=question

    endpoint=f"/triplets/link/{graph_name}/{head}/{tail}"
    response=requests.get(f"{base_url}{endpoint}")

    if response.status_code == 200:
        data = response.json()
        for i in range(len(data)):
            print(data[i]["relationship"])
    else:
        print(f"Error: {response.status_code}")


if __name__=="__main__":
    question=("Q10819","P102")
    infer_tail(question)