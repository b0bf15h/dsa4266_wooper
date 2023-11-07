import pathlib
import pandas as pd
from imblearn.over_sampling import SMOTENC


class SMOTESampler(object):
    """
    Performs SMOTE using the SMOTE NC algorithm from imblearn
    """

    def __init__(self, data_path: pathlib.Path, data: pd.DataFrame):
        """
        Initialize the Sampler object
        """
        self.data_path = data_path
        self.data = data
        self.labels = None
        self.features = None

    def SMOTE(self) -> pd.DataFrame:
        """
        Drop irrelevant categorical features before oversampling, returns dataframe containing features and labels
        """
        cols_to_drop = [
            "transcript_position",
            "transcript_length",
            "transcript_id",
            "gene_id",
        ]
        for col in cols_to_drop:
            if col in self.data.columns:
                self.data.drop(col, axis=1, inplace=True)
        truncated = self.data
        self.labels = truncated["label"]
        self.features = truncated.drop(["label"], axis=1)
        sm = SMOTENC(
            random_state=42, categorical_features=["sequence", "p1_seq", "m1_seq"]
        )
        self.features, self.labels = sm.fit_resample(self.features, self.labels)
        self.features["label"] = self.labels
        return self.features

    def write_output(self, df_name: str = 'balanced_train.pkl'):
        self.features["label"] = self.labels
        self.features.to_pickle(self.data_path / df_name)
        print("length of balanced training data: ")
        print(len(self.features))


# if __name__ == "__main__":
#     # df = pd.read_pickle()
#     # Oversampler = SMOTESampler(pathlib.Path(__file__).resolve().parents[1] / "data", df)
#     # Oversampler.SMOTE()
#     # Oversampler.write_output()