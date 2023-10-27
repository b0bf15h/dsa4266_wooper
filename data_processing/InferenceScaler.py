import pandas as pd
from sklearn.preprocessing import StandardScaler
from data_processing.SMOTE import SMOTESampler
import pickle
import pathlib


class InferenceProcessor(object):
    """
    Prepare data for inference based on transformations performed during model training
    """

    def __init__(
        self,
        data_path: pathlib.Path,
        scaler_name: str,
        encoder_name: str,
        output_filename: str,
    ):
        """
        Initialise object and read in data, encoder, scaler
        """
        self.data_path = data_path
        self.output_filename = output_filename
        self.df = pd.read_pickle(self.data_path / self.output_filename)
        self.scaler = self.get_scaler(scaler_name)
        self.encoder = self.get_encoder(encoder_name)
        self.reference = None

    def get_scaler(self, name: str):
        with open(self.data_path / name, "rb") as pickle_file:
            scaler = pickle.load(pickle_file)
        return scaler

    def get_encoder(self, name: str):
        with open(self.data_path / name, "rb") as pickle_file:
            encoder = pickle.load(pickle_file)
        return encoder

    def drop_columns(self):
        """
        Drops non-informative columns, saves id and position for indexing purposes
        """
        self.reference = self.df[["transcript_id", "transcript_position"]]
        self.df.drop(["transcript_id", "transcript_position"], axis=1, inplace=True)
        if "transcript_length" in self.df.columns:
            self.df.drop(["transcript_length"], axis=1, inplace=True)

    def scale(self):
        """
        Scale every float feature other than relative sequence position,
        which is already normalised
        """
        numeric_cols = self.df.select_dtypes(include=[float])
        if "relative_sequence_position" in numeric_cols.columns:
            numeric_cols.drop(["relative_sequence_position"], axis=1, inplace=True)
        self.df[numeric_cols.columns] = self.scaler.transform(numeric_cols)

    def encode(self):
        """
        Encode the 3 categorical features via OHE
        """
        encoded_data = self.encoder.transform(self.df[["sequence", "m1_seq", "p1_seq"]])
        encoded_column_names = self.encoder.get_feature_names_out(
            ["sequence", "m1_seq", "p1_seq"]
        )
        encoded_df = pd.DataFrame(encoded_data, columns=encoded_column_names)
        self.df = self.df.reset_index(drop=True)
        result_df = pd.concat([self.df, encoded_df], axis=1)
        result_df.drop(columns=["sequence", "m1_seq", "p1_seq"], inplace=True)
        self.df = result_df

    def write_output(self):
        print("Done processing inference data")
        self.df.to_pickle(self.data_path / self.output_filename)
        index = self.output_filename[0:8] + "_ids_and_positions.pkl"
        self.reference.to_pickle(self.data_path / index)
