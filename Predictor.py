from sklearn.ensemble import HistGradientBoostingClassifier
import pickle
from pathlib import Path
import pandas as pd

class Predictor(object):
    def __init__(self, model_path:Path, data_path:Path, data_name:str):
        self.model_path = model_path
        self.data_path = data_path
        self.data_name = data_name
        self.model = None
        self.data = None
        self.has_rp = False
        self.probs = None
        self.index = self.get_index()
    def get_index(self):
        if self.data_name.endswith('.pkl'):
            index_name = self.data_name[0:-4]+'_ids_and_positions.pkl'
        with open(self.data_path/ index_name, 'rb') as f:
            id = pickle.load(f)
        print('Index retrieved successfully')
        return id
    def get_model(self):
        if self.has_rp: 
            with open(self.model_path/'rf_final_model.pkl', 'rb') as f:
                m = pickle.load(f)
                print('Model with relative position retrieved successfully')
                self.model = m
            return
        with open(self.model_path/'rf_no_rp_final_model.pkl','rb') as f:
            m = pickle.load(f)
            print('Model without relative position retrieved successfully')
            self.model = m
        return
    def get_data(self):
        with open(self.data_path/ self.data_name, 'rb') as f:
            d = pickle.load(f)
        if 'relative_sequence_position' in d.columns:
            self.has_rp = True
        self.data = d
        print(len(self.data))
        print('Data retrieved successfully')
        return
    def drop_unused_cols(self):
        cols_to_drop = ['ensembl_gene_id',
       'start_position', 'end_position', 'strand', 'transcription_start_site',
       'transcript_count', 'percentage_gene_gc_content', 'gene_biotype',
       'transcript_biotype']
        for col in cols_to_drop:
            if col in self.data.columns:
                self.data.drop(columns=col, inplace=True)
        print(len(self.data))
    def drop_na_rows(self):
        self.data.dropna(inplace=True)
        print(len(self.data))
    def predict_probs (self):
        pairs = self.model.predict_proba(self.data)
        probs = [pair[1] for pair in pairs]
        self.probs = probs
        print("Predictions have been made")
    def write_output(self):
        self.index['score'] = self.probs
        if self.data_name.endswith('.pkl'):
            data_name = self.data_name[0:-4]
        output_fname = data_name + '_probs.csv'
        self.index.to_csv(self.data_path/output_fname, index = False)
        print("Predictions written to data path")
             