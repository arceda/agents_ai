from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import os
import json

# en windows es necesario especificar la ruta del .env
load_dotenv()
client = OpenAI()

response = client.moderations.create(
    model="omni-moderation-latest",
    input="are you stupid",
)


items = response.results[0].categories.__dict__.items()
detections = []
unsafe = False
if response.results[0].categories.illicit: 
    detections.append("illicit")
    unsafe = True
for e in items:
    if(e[1] == True):
        detections.append(e[0])
        unsafe = True

if unsafe:
    print("Guardrail  - " + ", ".join(detections) +  " content detected")

#response_dict = response.__str__()
#d = json.loads(response_dict)
#print(d)


##print(response.results[0].categories, "\n")
#print(vars(response.results[0].categories), "\n")
#print(response.results[0].categories.__dict__.items()) # ignora  atributos illicit y illicit/violence
#print(response.results[0].categories.illicit)
#print(response.results[0].categories.illicit)

#print(response_dict["results"][0].categories.__str__())
#d = json.loads(response_dict["results"][0].categories.__str__())
#print(d)

# se tuvo que hacer esto xq el atributo illlicit y illicit/violent no se podian parsear
cats = response.results[0].categories.__str__()
cats = cats.replace("Categories(", "")
cats = cats.replace(")", "")
print(cats.split(", "))


