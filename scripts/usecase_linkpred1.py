# 3. Hacer una predicción de tail en base a las entidades de una noticia y recomendar las noticias que contengan las entidades con mayor score en la predicción?

import requests
from pykeen.nn.utils import WikidataCache
import re
from typing import Sequence, Literal, List, Optional, cast

base_url="http://localhost:8000"
#graph_name=wikidata5m
graph_name="codexlarge"

GENERIC_RELATIONS=["P31","P463","P101"] #instance of, member of, field of work

wikidatacache=WikidataCache()

def get_links(entities):

    links=[]
    
    for i in range(len(entities)):
        for j in range(len(entities)):

            endpoint=f"/triplets/link/{graph_name}/{entities[i]}/{entities[j]}"
            response=requests.get(f"{base_url}{endpoint}")

            if response.status_code == 200:
                data = response.json()
                links+=[data[0]["relationship"]]

            else:
                print(f"Error: {response.status_code}")
                print(f"Las entidades {entities[i]} y/o {entities[j]} no están en {graph_name}(?)")
                return None
            
    return links

def get_tail(entity,link):

    endpoint=f"/triplets/entity/{graph_name}/{entity}/{link}"
    response=requests.get(f"{base_url}{endpoint}")

    if response.status_code == 200:
        data=response.json()
        tail=data[0]["tail"]

    else:
        print(f"Error: {response.status_code}")
        print(f"La entidad {entity} y/o {entity} no está en {graph_name}(?)")
        return None
    
    return tail


if __name__=="__main__":
    #entities=["Q312","Q95"] #Apple Google
    entities=["Q312","Q95"]
    links=get_links(entities)
    print(links)

    for entity in entities:
        for link in links:
            
            tail_label=wikidatacache.get_labels([get_tail(entity,link)])
            entity_label=wikidatacache.get_labels([entity])

            print(f"HEAD:{entity_label}, RELATIONSHIP:{link}, TAIL:{tail_label}".format(entity_label, link, tail_label))



