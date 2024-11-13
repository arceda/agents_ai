import numpy as np
import pandas as pd
import pandas as pd
import numpy as np

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

def get_metrics(y_test, y_test_predictions):
    accuracy = accuracy_score(y_test, y_test_predictions)
    precision = precision_score(y_test, y_test_predictions)
    recall = recall_score(y_test, y_test_predictions)
    f1score = f1_score(y_test, y_test_predictions)      

    return {"accuracy":accuracy, "precision":precision, "recall":recall, "f1score":f1score}

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

data = pd.read_csv("results_part_1.csv")
data_filtered = pd.DataFrame()

data_filtered['text'] = data['text']
data_filtered['label'] = data['label']

acc = []
precision = []
recall = []
f1 = []


for model in models:
    data_filtered[model] = data["codes-"+model]
    data_filtered[model] = np.where(data_filtered[model]==400, 1, 0)

    metrics = get_metrics(data_filtered["label"], data_filtered[model])

    acc.append(metrics["accuracy"])
    precision.append(metrics["precision"])
    recall.append(metrics["recall"])
    f1.append(metrics["f1score"])


    
metrics_pd = pd.DataFrame()
metrics_pd["model"] = models
metrics_pd["acc"] = acc
metrics_pd["precision"] = precision
metrics_pd["recall"] = recall
metrics_pd["f1-score"] = f1

print(data.head())

print(metrics_pd)
# chatgpt-4o es el mejor

tmp = data_filtered[( data_filtered.label != data_filtered["gpt-4o-mini"]) ]
new_data = pd.DataFrame()
new_data["text"] = tmp["text"]
new_data["label"] = tmp["label"]
new_data["gpt-4o-mini"] = tmp["gpt-4o-mini"]


new_data.to_csv("bad_samples.csv")


    