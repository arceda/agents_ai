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

data = pd.read_csv("results_.csv")
data_filtered = pd.DataFrame()

data_filtered['text'] = data['text']
data_filtered['label'] = data['label']

for model in models:
    data_filtered[model] = data["codes"+model]
    data_filtered[model] = np.where(data_filtered[model]==400, 1, 0)
    print(model, get_metrics(data_filtered["label"], data_filtered[model]))