import optuna
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

data_dir = Path(__file__).resolve().parents[1]/'data'
random_state = 42
n_jobs = -1
def objective(trial, X_train, y_train, X_valid, y_valid):
    n_estimators  = trial.suggest_int('n_estimators', 150,700)
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
    ccp_alpha = trial.suggest_float('ccp_alpha', 0.0, 0.5)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
    min_samples_split = trial.suggest_int('min_samples_split', 2, np.floor(0.01*len(X_train)))
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, np.floor(0.005*len(X_train)))
    rf = RandomForestClassifier(n_estimators = n_estimators, criterion = criterion, ccp_alpha = ccp_alpha,
                                max_features = max_features, min_samples_split = min_samples_split,
                                min_samples_leaf =  min_samples_leaf, random_state = random_state, n_jobs = n_jobs)
    rf.fit(X_train, y_train)
    probs = rf.predict_proba(X_valid)
    true_probs = [entry[1] for entry in probs]
    return roc_auc_score(y_true=y_valid,y_score=true_probs)

def extract_df(df):
    return df.drop(['label'], axis = 1), df['label']

def get_data():
    train = pd.read_pickle(data_dir/'train_OHE.pkl')
    trainx, trainy = extract_df(train)
    validation = pd.read_pickle(data_dir/'validation_OHE.pkl')
    validationx, validationy = extract_df(validation)
    trainf = pd.read_pickle(data_dir/'train_final_OHE.pkl')
    trainfx, trainfy = extract_df(trainf)
    test = pd.read_pickle(data_dir/'test_final_OHE.pkl')
    testx, testy = extract_df(test)
    return trainx, trainy, validationx, validationy, trainfx, trainfy, testx, testy

def tune_model(trainx, trainy, validx, validy, n_trials):
    study = optuna.create_study(study_name = 'random_forest',pruner = optuna.pruners.HyperbandPruner(
        min_resource=1, reduction_factor=3
    ), direction = "maximize")
    study.optimize(lambda trial: objective(trial, trainx, trainy, validx, validy), n_trials= n_trials, n_jobs= -1)
    with open(data_dir/'rf_params.pkl', 'wb') as pickle_file:
        pickle.dump(study.best_params, pickle_file)

if __name__ == "__main__":
    trainx, trainy, validationx, validationy, trainfx, trainfy, testx, testy = get_data()
    tune_model(trainx, trainy, validationx, validationy, 50)