import optuna
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

data_dir = Path(__file__).resolve().parents[1]/'data'
random_state = 42
n_jobs = -1
def objective(trial, X_train, y_train, X_valid, y_valid):
    layer1 = trial.suggest_int('layer1', 60,140)
    layer2 = trial.suggest_int('layer2', 10,90)
    alpha = trial.suggest_float('alpha', 1e-7,1e-2, log = True)
    max_iter = trial.suggest_int('max_iter', 200, 350, step = 3)
    nn = MLPClassifier(hidden_layer_sizes = (layer1, layer2), alpha = alpha, max_iter = max_iter, early_stopping = True, random_state = random_state)
    nn.fit(X_train, y_train)
    probs = nn.predict_proba(X_valid)
    true_probs = [entry[1] for entry in probs]
    return roc_auc_score(y_true=y_valid,y_score=true_probs)

def extract_df(df):
    return df.drop(['label'], axis = 1), df['label']

def get_data(train: str = 'train_OHE.pkl', validate: str = 'validation_OHE.pkl' , trainf: str = 'train_final_OHE.pkl', test:str = 'test_final_OHE.pkl'):
    train = pd.read_pickle(data_dir/train)
    trainx, trainy = extract_df(train)
    validation = pd.read_pickle(data_dir/validate)
    validationx, validationy = extract_df(validation)
    trainf = pd.read_pickle(data_dir/trainf)
    trainfx, trainfy = extract_df(trainf)
    test = pd.read_pickle(data_dir/test)
    testx, testy = extract_df(test)
    return trainx, trainy, validationx, validationy, trainfx, trainfy, testx, testy

def tune_model(trainx, trainy, validx, validy, n_trials):
    study = optuna.create_study(study_name = 'MLP',pruner = optuna.pruners.HyperbandPruner(
        min_resource=1, reduction_factor=3
    ), direction = "maximize")
    study.optimize(lambda trial: objective(trial, trainx, trainy, validx, validy), n_trials= n_trials, n_jobs= n_jobs)
    with open(data_dir/'MLP_params.pkl', 'wb') as pickle_file:
        pickle.dump(study.best_params, pickle_file)

if __name__ == "__main__":
    trainx, trainy, validationx, validationy, trainfx, trainfy, testx, testy = get_data()
    tune_model(trainx, trainy, validationx, validationy, 100)