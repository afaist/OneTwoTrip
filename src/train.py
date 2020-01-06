import os
from sklearn import preprocessing
import pandas as pd
from sklearn import ensemble, metrics
import joblib

from . import dispatcher

TARGET = 'goal1'

TRAINING_DATA = os.environ.get("TRAINING_DATA")
FOLD = int(os.environ.get("FOLD"))
print(f"FOLD = {FOLD}")
MODEL = os.environ.get("MODEL")

FOLD_MAPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}

if __name__ == "__main__":
    df = pd.read_csv(TRAINING_DATA)
    train_df = df[df["kfold"].isin(FOLD_MAPPING.get(FOLD))]
    valid_df = df[df.kfold == FOLD]

    ytrain = train_df[TARGET].values
    yvalid = valid_df[TARGET].values

    drop_columns = list(filter(lambda x: 'goal' in x, train_df.columns))
    drop_columns.append('userid')
    drop_columns.append('orderid')
    drop_columns.append('kfold')

    train_df = train_df.drop(drop_columns, axis=1)
    valid_df = valid_df[train_df.columns]

#    label_encoders = []
#    for c in train_df.columns:
#        lbl = preprocessing.LabelEncoder()
#        lbl.fit(train_df[c].values.tolist() + valid_df[c].values.tolist())
#        train_df.loc[:, c] = lbl.transform(train_df[c].values.tolist())
#        valid_df.loc[:, c] = lbl.transform(valid_df[c].values.tolist())
#        label_ecnoders.append((c, lbl))

# data is ready to train
    clf = dispatcher.MODELS[MODEL]
    clf.fit(train_df, ytrain)
    print(clf)
 
    preds = clf.predict_proba(valid_df)[:, 1]
    print(metrics.roc_auc_score(yvalid, preds))

    joblib.dump(clf, f"models/{MODEL}.pkl")