# Detección de triplas verdaderas y falsas

import requests

base_url="http://localhost:8000"
#graph_name="dbpedia50"
graph_name="nations"

threshold=-1.5

triple=("israel","violentactions","egypt")


def classify_triple(threshold,triple):

    head,relationship,tail=triple
    endpoint=f"/triplets/probability/{graph_name}/{head}/{relationship}/{tail}"

    response=requests.get(f"{base_url}{endpoint}")

    if response.status_code == 200:
        data = response.json()
        print("threshold:",threshold)
        print("Score:",data["score"])
        print(f"Triple {triple} is",data["score"]>threshold)

    else:
        print(f"Error: {response.status_code}")


if __name__=="__main__":
    classify_triple(threshold,triple)