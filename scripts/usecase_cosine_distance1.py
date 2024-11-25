# 1. Dada una noticia que contiene x entidades, calcular las entidades más cercanas en el espacio vectorial,
# recomendar las noticias que las contengan.

import requests
from pykeen.nn.utils import WikidataCache

base_url="http://localhost:8000"
#graph_name=wikidata5m
graph_name="codexlarge"

wikidatacache=WikidataCache()

def get_k_closest_entities(entities,k=10):

    for entity in entities:

        endpoint=f"/closest_entities/{graph_name}/{entity}"
        response=requests.get(f"{base_url}{endpoint}")

        if response.status_code == 200:
            data = response.json()

            for res_entity in data:
                print(wikidatacache.get_labels(wikidata_identifiers=[res_entity["name"]]),"\t SIMILARITY:",res_entity["similarity"])
        #elegir subgrupo?
        #
        else:
            print(f"Error: {response.status_code}")
            print(f"La entidad {entity} no está en {graph_name}(?)")

    return None



if __name__=="__main__":
    
    # #example 1 VIDEOGAMES (con wikidata5m)
    # entities=["Q1323662","Q41187", "Q132020"] #Playstation, Sony y Xbox
    # print(get_k_closest_entities(entities))

    #example 2 POLITICS (con codexlarge)
    # entities=["Q22686","Q10819"] #Donald Trump y Mariano Rajoy
    # get_k_closest_entities(entities)

    #example 3 SCIENCE AND TECHNOLOGY (codexlarge)
    entities=["Q95","Q2766"]
    get_k_closest_entities(entities)
