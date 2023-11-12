from sklearn.ensemble import RandomForestClassifier
import pickle
from pathlib import Path
import pandas as pd

class TrainModel(object):
    def __init__(self, model_path:Path, data_path:Path, params_name:str, data_name:str, model_name:str):
        self.model_path = model_path
        self.data_path = data_path
        self.model_params = self.parse_params(params_name)
        self.data = None
        self.model = None
        self.data_name = data_name
        self.model_name = model_name
    def get_best_params(self, lparams):
        params = {}
        max = -1
        for trial in lparams:
            if sum(trial.values) >max:
                max = sum(trial.values)
                params = trial.params
        return params
    def drop_rp(self):
        self.data.drop(['relative_sequence_position'], axis = 1, inplace = True)
    def parse_params(self, param_name:str):
        with open(self.model_path/param_name, 'rb') as f:
            params = pickle.load(f)
        params = self.get_best_params(params)
        return params
    def get_data(self):
        with open(self.data_path/self.data_name, 'rb') as f:
            data = pickle.load(f)
        self.data = data
    def extract_df(self, df):
        return df.drop(['label'], axis = 1), df['label']
    def train_model(self):
        trainx, trainy = self.extract_df(self.data)
        rf = RandomForestClassifier(n_estimators = self.model_params.get('n_estimators'), criterion = self.model_params.get('criterion'), 
                                    max_features = self.model_params.get('max_features'), min_samples_split = self.model_params.get('min_samples_split'),
                                    min_samples_leaf = self.model_params.get('min_samples_leaf'), random_state= 42, n_jobs = -1)
        rf.fit(trainx, trainy)
        self.model = rf
    def write_model(self):
        with open(self.model_path/self.model_name, 'wb') as f:
            pickle.dump(self.model,f)
        