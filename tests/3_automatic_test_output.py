# evalua el desempeño de los GurdRails sobre una DB. Solo usa gpt-4o-mini
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
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
from argparse import ArgumentParser

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    auc, roc_curve,
    matthews_corrcoef
)
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

# example: python 3_automatic_test_output.py -i datasets/urls.csv -o1 urls_results.csv -o2 urls_bad.csv 

parser = ArgumentParser()
parser.add_argument("-i", "--input", help = "dataset of prompt injections .csv")
parser.add_argument("-o1", "--output1", help = "API calls result for each prompt .csv")
parser.add_argument("-o2", "--output2", help = "API calls that pass the guardrail .csv")
args = parser.parse_args()

print(args)

input_file = args.input
output_file_1 = args.output1
output_file_2 = args.output2


#model = "gpt-4o-mini"
model = "vertex_ai/claude-3-opus-20240229"

url = "http://localhost:4000/chat/completions" # with litellm
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer sk-1234"
}

def get_metrics(y_test, y_test_predictions):
    accuracy = accuracy_score(y_test, y_test_predictions)
    precision = precision_score(y_test, y_test_predictions)
    recall = recall_score(y_test, y_test_predictions)
    f1score = f1_score(y_test, y_test_predictions)      

    return {"accuracy":accuracy, "precision":precision, "recall":recall, "f1score":f1score}

def call_LLM(prompt, llm):
    data = {
        "model": llm,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assitant." 
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "guardrails": ["llm-output-guard"]
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    return response.status_code, response.text 
    

def evaluate():
    data = pd.read_csv(input_file)
    #print(data)

    texts = data["text"]
    
    codes = []
    responses = []
    for i, text in enumerate(texts):  
        print("testing... " + str(i) + " " + text)
    

        code, result = call_LLM(text, model)       
        codes.append(code)
        responses.append(result)   
    
    data["codes-"+model] = codes
    data["responses-"+model] = responses
    
    #print(data)
    data.to_csv(output_file_1, index=False)


# api calls
evaluate()
print("finish calling liteLLM. Result save in:", output_file_1)

# analize results
data = pd.read_csv(output_file_1)
data["label"] = np.where(data["class"]=="benign", 0, 1) # # la BD de URL tiene varias clases
data[model] = np.where(data["codes-"+model]==400, 1, 0) # convertimos a 1 o 0
metrics = get_metrics(data["label"], data[model])

print(data)
print(metrics)


tmp = data[( data.label != data[model]) ]
new_data = pd.DataFrame()
new_data["text"] = tmp["text"]
new_data["label"] = tmp["label"]
new_data[model] = tmp[model]

print("finish analysis, prompt that passes the guardrail are store here:", output_file_2)
new_data.to_csv(output_file_2)

