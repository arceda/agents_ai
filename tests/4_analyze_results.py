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


def get_metrics(y_test, y_test_predictions):
    accuracy = accuracy_score(y_test, y_test_predictions)
    precision = precision_score(y_test, y_test_predictions)
    recall = recall_score(y_test, y_test_predictions)
    f1score = f1_score(y_test, y_test_predictions)      

    return {"accuracy":accuracy, "precision":precision, "recall":recall, "f1score":f1score}

model = "gpt-4o-mini"

data = pd.read_csv("custom_results.csv")
data[model] = np.where(data["codes-"+model]==400, 1, 0) # convertimos a 1 o 0
metrics = get_metrics(data["label"], data[model])
print(metrics)


tmp = data[( data.label != data["gpt-4o-mini"]) ]
new_data = pd.DataFrame()
new_data["text"] = tmp["text"]
new_data["label"] = tmp["label"]
new_data["gpt-4o-mini"] = tmp["gpt-4o-mini"]

new_data.to_csv("custom_bad_samples.csv")


    