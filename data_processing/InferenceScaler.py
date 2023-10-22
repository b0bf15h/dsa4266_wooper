import pandas as pd
from sklearn.preprocessing import StandardScaler
from data_processing.SMOTE import SMOTESampler
import pickle

class InferenceProcessor(object):
    def __init__(self,data_path,scaler,encoder,output_filename):
        self.data_path = data_path
        self.output_filename = output_filename
        self.df = pd.read_pickle(self.data_path/self.output_filename)
        self.scaler = scaler
        self.encoder = encoder
        self.get_scaler()
        self.get_encoder()
        self.reference = None
    def get_scaler(self):
        with open(self.data_path/self.scaler, 'rb') as pickle_file:
            self.scaler = pickle.load(pickle_file)
    def get_encoder(self):
        with open(self.data_path/self.encoder, 'rb') as pickle_file:
            self.encoder = pickle.load(pickle_file)
    def drop_columns(self):
        self.reference = self.df[['transcript_id', 'transcript_position']]
        self.df.drop(['transcript_id', 'transcript_position'], axis = 1, inplace = True)
        if 'transcript_length' in self.df.columns:
            self.df.drop(['transcript_length'], axis = 1, inplace = True)
    def scale(self):
        numeric_cols = self.df.select_dtypes(include=[float])
        if 'relative_sequence_position' in numeric_cols.columns:
            numeric_cols.drop(['relative_sequence_position'], axis = 1, inplace = True)
        self.df[numeric_cols.columns] = self.scaler.transform(numeric_cols)
    def encode(self):
        encoded_data = self.encoder.transform(self.df[["sequence", "m1_seq", "p1_seq"]])
        encoded_column_names = self.encoder.get_feature_names_out(["sequence", "m1_seq", "p1_seq"])
        encoded_df = pd.DataFrame(encoded_data, columns=encoded_column_names)
        self.df = self.df.reset_index(drop=True)
        result_df = pd.concat([self.df, encoded_df], axis=1)
        result_df.drop(columns=["sequence", "m1_seq", "p1_seq"], inplace=True)
        self.df = result_df
    def write_output(self):
        print("Done processing inference data")
        self.df.to_pickle(self.data_path/self.output_filename)
        self.reference.to_pickle(self.data_path/'ids_and_positions.pkl')