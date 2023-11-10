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
        data_path: Path,
        train_final: str = 'train_final.pkl',
        test_final: str = 'test_final.pkl',
        train: str = 'train.pkl',
        validation: str = 'validation.pkl'
    ):
        self.data_path = Path(data_path)
        self.train_final = pd.read_pickle(self.data_path / train_final)
        self.test_final = None
        self.validation = None
        self.train = None
        if test_final != '':
            self.test_final = pd.read_pickle(self.data_path / test_final)
        if train != '':
            self.train = pd.read_pickle(self.data_path / train)
        if validation != '':
            self.validation = pd.read_pickle(self.data_path / validation)

    def one_hot_encode(self, df: pd.DataFrame, encoder_available: bool = False, encoder_name: str = 'OHE.pkl'):
        """Performs OneHot Encoding on the df, and writes the used encoder for future use if no encoders are available"""
        if not encoder_available:
            enc = preprocessing.OneHotEncoder(
                sparse_output = False, drop = "if_binary", handle_unknown="ignore"
            )
            encoded_data = enc.fit_transform(df[["sequence", "m1_seq", "p1_seq"]])
            with open(self.data_path / encoder_name, "wb") as pickle_file:
                pickle.dump(enc, pickle_file)
        else:
            with open(self.data_path / encoder_name, "rb") as pickle_file:
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
        """OHE all 4 DFs if possible"""
        self.train_final = self.one_hot_encode(self.train_final, False)
        if isinstance(self.test_final, pd.DataFrame):
            self.test_final = self.one_hot_encode(self.test_final, True)
        if isinstance(self.train, pd.DataFrame):
            self.train = self.one_hot_encode(self.train, True)
        if isinstance(self.validation, pd.DataFrame):
            self.validation = self.one_hot_encode(self.validation, True)


    def write_output(self, trainf: str = 'train_final_OHE.pkl', testf: str = 'test_final_OHE.pkl', train: str = 'train_OHE.pkl', validation: str = 'validation_OHE.pkl'):
        self.train_final.to_pickle(self.data_path / trainf)
        if isinstance(self.test_final, pd.DataFrame):
            self.test_final.to_pickle(self.data_path / testf)
        if isinstance(self.train, pd.DataFrame):
            self.train.to_pickle(self.data_path / train)
        if isinstance(self.validation, pd.DataFrame):
            self.validation.to_pickle(self.data_path / validation)
