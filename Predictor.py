from sklearn.ensemble import HistGradientBoostingClassifier
import pickle
from pathlib import Path
import pandas as pd

class Predictor(object):
    def __init__(self, model_path:Path, data_path:Path, model_name, data_name):
        self.model_name = model_name
        self.model_path = model_path
        self.data_path = data_path
        self.data_name = data_name
        self.model = self.get_model()
        self.data = self.get_data()
        self.probs = None
    def get_model(self):
        with open(self.model_path/ self.model_name, 'rb') as f:
            m = pickle.load(f)
        print('Model retrieved successfully')
        return m
    def get_data(self):
        with open(self.data_path/ self.data_name, 'rb') as f:
            d = pickle.load(f)
        print('Data retrieved successfully')
        return d
    def drop_unused_cols(self):
        cols_to_drop = ['ensembl_gene_id',
       'start_position', 'end_position', 'strand', 'transcription_start_site',
       'transcript_count', 'percentage_gene_gc_content', 'gene_biotype',
       'transcript_biotype']
        for col in cols_to_drop:
            if col in self.data.columns:
                self.data.drop(columns=col, inplace=True)
    def drop_na_rows(self):
        self.data = self.data.dropna()
    def predict_probs (self):
        pairs = self.model.predict_proba(self.data)
        probs = [pair[1] for pair in pairs]
        self.probs = probs
        print("Predictions have been made")
    def write_output(self):
        output_fname = self.data_name + '_probs.pkl'
        with open(self.data_path/output_fname, 'wb') as file:
            pickle.dump(self.probs, file)
        print("Predictions written to data path")
             