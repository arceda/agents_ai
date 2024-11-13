# evalua el desempeño de los GurdRails sobre una DB. Hace la comparación con varios LLMs
# author: Vicente

import json
import requests
import numpy as np
import glob
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import os
import pandas as pd


models = [
    "gpt-3.5-turbo-1106",
    "gpt-4o",
    "gpt-4o-mini",

    "vertex_ai/claude-3-5-haiku-20241022",
    "vertex_ai/claude-3-5-sonnet-v2-20241022",
    "vertex_ai/claude-3-opus-20240229",

    "gemini-1.5-flash",
    "gemini-1.5-flash-002",
    "gemini-1.5-pro",
    "gemini-1.5-pro-002",

    "awsbedrock/meta.llama3-2-90b",
    "awsbedrock/meta.llama3-2-3b"
]

url = "http://localhost:4000/chat/completions" # with litellm
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer sk-1234"
}

def call_LLM(prompt, llm):
    data = {
        "model": llm,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "guardrails": ["prompt-injection-during-guard"]
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    return response.status_code, response.text 
    #print(response.status_code)
    #print(response.text)


    # Check if the request was successful
    #if response.status_code == 200:
        # Print the response JSON if the request was successful
    #    print("Response:", json.dumps(response.json(), indent=2))
    #else:
        # Print the error message if the request was unsuccessful
    #    print("Failed with status code:", response.status_code)
    #    print("Error message:", response.text)
    


def evaluate():
    data = pd.read_csv('datasets/small_test_part_1.csv')
    #data = pd.read_parquet('datasets/train.parquet', engine='pyarrow')

    texts = data["text"]
    

    for i, model in enumerate(models):
        print("\n", i, "iteration...")
        codes = []
        responses = []
        for text in texts:  
            print("testing", text, model)
            code, result = call_LLM(text, model)       
            codes.append(code)
            responses.append(result)   
        
        data["codes-"+model] = codes
        data["responses-"+model] = responses
    

    data.to_csv("results_part_1.csv", index=False)

evaluate()


