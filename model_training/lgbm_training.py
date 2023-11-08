import optuna.integration.lightgbm as lgb
from lightgbm import early_stopping
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.metrics import roc_auc_score, average_precision_score

data_dir = Path(__file__).resolve().parents[1]/'data'
random_state = 42
n_jobs = -1


new_params = {
    "objective": "binary",
    "metric": ["auc", "average_precision"],
    "verbosity": -1,
    "boosting_type": "gbdt",
}

def extract_df(df):
    return df.drop(['label'], axis = 1), df['label']

def get_data(train: str = 'train_OHE.pkl', validate: str = 'validation_OHE.pkl' , trainf: str = 'train_final_OHE.pkl', test:str = 'test_final_OHE.pkl'):
    import lightgbm as lgbm
    train = pd.read_pickle(data_dir/train)
    trainx, trainy = extract_df(train)
    dtrain = lgbm.Dataset(trainx, label=trainy)
    validation = pd.read_pickle(data_dir/validate)
    validationx, validationy = extract_df(validation)
    dvalidation = lgbm.Dataset(validationx, label=validationy)
    trainf = pd.read_pickle(data_dir/trainf)
    trainfx, trainfy = extract_df(trainf)
    dtrainf = lgbm.Dataset(trainfx, label = trainfy)
    test = pd.read_pickle(data_dir/test)
    testx, testy = extract_df(test)
    return dtrain, dvalidation, dtrainf, testx, testy

def tune_model(dtrain, dval, trials:int = 1000):
    model = lgb.train(
        new_params,
        dtrain,
        valid_sets=[dtrain, dval],
        callbacks=[early_stopping(75)],
        num_boost_round = trials,
        optuna_seed = random_state
    )
    with open(data_dir/'lgbm_params.pkl', 'wb') as f:
        pickle.dump(model.params, f)
        
if __name__ == '__main__':
    dtrain, dvalidation, dtrainf, testx, testy = get_data()
    tune_model(dtrain, dvalidation, 1200)