import optuna
import pandas as pd
from pathlib import Path
import pickle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score

data_dir = Path(__file__).resolve().parents[1]/'data'
random_state = 42
n_jobs = -1
def objective(trial, X_train, y_train, X_valid, y_valid):
    n_estimators = trial.suggest_int('n_estimators', 50,150, step = 4)
    learning_rate = trial.suggest_float('learning_rate',0.2, 20, log = True)
    ada = AdaBoostClassifier(n_estimators = n_estimators, learning_rate = learning_rate, random_state = random_state)
    ada.fit(X_train, y_train)
    probs = ada.predict_proba(X_valid)
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
    study = optuna.create_study(study_name = 'adaboost',pruner = optuna.pruners.HyperbandPruner(
        min_resource=1, reduction_factor=3
    ), direction = "maximize")
    study.optimize(lambda trial: objective(trial, trainx, trainy, validx, validy), n_trials= n_trials, n_jobs= n_jobs)
    with open(data_dir/'adaboost_params.pkl', 'wb') as pickle_file:
        pickle.dump(study.best_params, pickle_file)

if __name__ == "__main__":  
    trainx, trainy, validationx, validationy, trainfx, trainfy, testx, testy = get_data()
    tune_model(trainx, trainy, validationx, validationy, 80)