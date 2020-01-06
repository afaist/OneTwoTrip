from sklearn import ensemble

MODELS = {
    "randomforest": ensemble.RandomForestClassifier(n_jobs=-1, verbose=2, n_estimators=300, random_state=42),
    "extratrees": ensemble.ExtraTreesClassifier(n_jobs=-1, verbose=2, n_estimators=300, random_state=2020)
}
