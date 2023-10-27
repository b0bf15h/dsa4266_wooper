from sklearn import preprocessing
import pandas as pd
import pickle
from pathlib import Path


class OneHotEncoder(object):
    """
    performs OHE on the respective dataframes
    """

    def __init__(
        self,
        train_final: str,
        test_final: str,
        train: str,
        validation: str,
        data_path: Path,
    ):
        self.data_path = data_path
        self.train_final = pd.read_pickle(self.data_path / train_final)
        self.test_final = pd.read_pickle(self.data_path / test_final)
        self.train = pd.read_pickle(self.data_path / train)
        self.validation = pd.read_pickle(self.data_path / validation)

    def one_hot_encode(self, df: pd.DataFrame, encoder_available: bool = False):
        """Performs OneHot Encoding on the df, and writes the used encoder for future use if no encoders are available"""
        file_name = "encoder.pkl"
        if not encoder_available:
            enc = preprocessing.OneHotEncoder(
                sparse=False, drop="first", handle_unknown="ignore"
            )
            encoded_data = enc.fit_transform(df[["sequence", "m1_seq", "p1_seq"]])
            with open(self.data_path / file_name, "wb") as pickle_file:
                pickle.dump(enc, pickle_file)
        else:
            with open(self.data_path / file_name, "rb") as pickle_file:
                enc = pickle.load(pickle_file)
            encoded_data = enc.transform(df[["sequence", "m1_seq", "p1_seq"]])
        encoded_column_names = enc.get_feature_names_out(
            ["sequence", "m1_seq", "p1_seq"]
        )
        encoded_df = pd.DataFrame(encoded_data, columns=encoded_column_names)
        df = df.reset_index(drop=True)
        result_df = pd.concat([df, encoded_df], axis=1)
        result_df.drop(columns=["sequence", "m1_seq", "p1_seq"], inplace=True)
        return result_df

    def OHE(self):
        """OHE all 4 DFs"""
        self.train_final = self.one_hot_encode(self.train_final)
        self.test_final = self.one_hot_encode(self.test_final, True)
        self.train = self.one_hot_encode(self.train, True)
        self.validation = self.one_hot_encode(self.validation, True)

    def write_output(self):
        self.train_final.to_pickle(self.data_path / "train_final_OHE.pkl")
        self.test_final.to_pickle(self.data_path / "test_final_OHE.pkl")
        self.train.to_pickle(self.data_path / "train_OHE.pkl")
        self.validation.to_pickle(self.data_path / "validation_OHE.pkl")
