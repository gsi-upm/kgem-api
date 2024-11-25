# 2. Dadas dos noticias (dos grupos de entidades) calcular la similitud entre ambas.

import requests
import numpy as np

base_url="http://localhost:8000"
#graph_name=wikidata5m
graph_name="codexlarge"


def get_total_similarity(entity_list1,entity_list2):

    similarity_matrix = np.zeros((len(entity_list1), len(entity_list2)))

    for i in range(len(entity_list1)):
        for j in range(len(entity_list2)):

            endpoint=f"/distance/{graph_name}/{entity_list1[i]}({entity_list2[j]})"
            response=requests.get(f"{base_url}{endpoint}")

            if response.status_code == 200:
                similarity_matrix[i, j] = response.json()["similarity"]

            else:
                print(f"Error: {response.status_code}")
                print(f"Las entidades {entity_list1[i]} y/o {entity_list2[j]} no están en {graph_name}(?)")
                return
    
    return similarity_matrix


if __name__=="__main__":
    
    # example 1: similitud entre dos noticias que no tienen nada que ver 
    # -'ISIS is the biggest beneficiary': Graham lashes out at Trump over Syria withdrawal
    # - This Sony PS5 feature could end the PlayStation vs Xbox battle once and for all
    list1=["PlayStation","Sony","Xbox (console)"]
    list2=["Donald Trump","American-led intervention in the Syrian Civil War","Islamic State of Iraq and the Levant"]
    print(get_total_similarity(list1,list2))

    # example 2: similitud entre dos noticias de videojuegos
    # - Zelda Fan Builds Enormous Wooden Map Of Hyrule
    # - Untitled Goose Game is the best selling game on Switch right now
    list1=["Untitled Goose Game","Nintendo Switch"]
    list2=["Universe of The Legend of Zelda"]
    print(get_total_similarity(list1,list2))

    #example 3: dada una noticia elegir la noticia con mayor similitud de entre un grupo de n noticias
       