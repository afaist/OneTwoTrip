import os
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn import ensemble, metrics
import joblib

from . import dispatcher

TARGET = 'goal1'
TEST_DATA = os.environ.get("TEST_DATA")


MODEL = os.environ.get("MODEL")


def predict():
    df = pd.read_csv(TEST_DATA)
    test_idx = df["orderid"]
    for FOLD in range(5):
        encoders = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}_label_encoder.pkl"))
        
        for c in encoders:
            lbl = encoders[c]
            df.loc[:, c] = lbl.transform(df[c].values.tolist())
            
        clf = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}.pkl"))
    
        cols = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}_columns.pkl"))
    
        preds = clf.predict_proba(df[cols])[:, 1]
        
        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds
            
    predictions /= 5
    sub = pd.DataFrame(np.column_stack((test_idx, predictions)), columns=['orderid','proba'])
    return sub

if __name__ == "__main__":
    submission = predict()
    submission.to_csv(f"models/{MODEL}.csv", index=False)