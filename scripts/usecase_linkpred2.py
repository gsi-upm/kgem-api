# 4. Determinar la relación que hay entre dos noticias y decidir si recomendarla.

import requests
from pykeen.nn.utils import WikidataCache
import re
from typing import Sequence, Literal, List, Optional, cast

base_url="http://localhost:8000"
#graph_name=wikidata5m
graph_name="codexlarge"

GENERIC_RELATIONS=["P31","P463","P101","P9219"] #instance of, member of, field of work, Discogs style ID

wikidatacache=WikidataCache()

def get_links(entities1,entities2):

    links=[]
    
    for i in range(len(entities1)):
        for j in range(len(entities2)):

            endpoint=f"/triplets/link/{graph_name}/{entities1[i]}/{entities2[j]}"
            response=requests.get(f"{base_url}{endpoint}")

            if response.status_code == 200:
                data = response.json()
                links+=[data[0]["relationship"]]

            else:
                print(f"Error: {response.status_code}")
                print(f"Las entidades {entities1[i]} y/o {entities2[j]} no están en {graph_name}(?)")
                return None
            
    return links

if __name__=="__main__":

    # Justin Bieber Jokingly Reenacts Taylor Swift's Banana Freak Out Moment After Lasik Surgery
    entities1=["Q26876","Q34086","Q278846"] #Taylor Swift, Justin Bieber, LASIK
    # 
    entities2=[]

    links=get_links(entities1,entities2)
    print(links)
