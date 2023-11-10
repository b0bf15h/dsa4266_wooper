from data_processing.DataProcessing import (
    DataParsing,
    SummariseDataByTranscript,
    MergeData,
)
from data_processing.SMOTE import SMOTESampler
from data_processing.TrainTestSplit import TrainTestSplit
from data_processing.Scaler import Scaler
from data_processing.OHE import OneHotEncoder
import pandas as pd
import argparse
import pathlib

parser = argparse.ArgumentParser(
    prog="main.py", description="End-to-End Model Training"
)
parser.add_argument(
    "--data_path",
    "-d",
    type=pathlib.Path,
    action="store",
    required=True,
    help="Path to raw data and labels",
)
parser.add_argument(
    "--train_test_ratio",
    "-r",
    type=float,
    action="store",
    required=False,
    help="Ratio for train-test split",
    default=0.8,
)
parser.add_argument(
    "--step",
    "-s",
    type=int,
    action="store",
    required=True,
    help="Step 1 is parsing json data and creating df, to be done in python. Step 2 requires output from R script which queries Biomart for additional features",
)
args = parser.parse_args()
data_path = args.data_path
tt_ratio = args.train_test_ratio
step = args.step


class WooperModel(object):
    def __init__(self, data_path):
        self.raw_data = []
        self.raw_info = []
        self.data_path = data_path
        self.df = None
        # self.reference = []

    # Task 1
    def basic_transformations(self, raw_data, raw_info):
        """Transforms zipped unlabelled json data of read level into labelled dataframe at sequence level, prepare for querying transcript length"""
        self.raw_data = self.data_path / raw_data
        self.raw_info = self.data_path / raw_info
        parsed_data = DataParsing(self.raw_data).unlabelled_data()
        summarised_data = SummariseDataByTranscript(parsed_data).summarise()
        MergeData(summarised_data, self.raw_info, self.data_path).write_data_for_R()
        # MergeData(parsed_data, self.raw_info, self.data_path).write_data_for_R()

    def advanced_transformations(self, csv_data: str = 'biomart_data.csv'):
        """
        Merges data with transcript length queried from R, as well as perform train,test split;
        oversampling; normalisation and encoding. Finalised dataframes are ready for use in model training and written into data_path
        """
        csv_data = self.data_path / csv_data
        step1_data = self.data_path / "interm.pkl"
        self.df = pd.read_pickle(step1_data)
        merger = MergeData(self.df, csv_data, self.data_path)
        merged_data = merger.merge_with_features()
        merged_data = merger.drop_unused_features(merged_data)
        # merged_data.to_pickle(self.data_path/'ds0_reads.pkl')
        train, _ = TrainTestSplit(merged_data).train_test_split(
            tt_ratio, data_path, "train_data.pkl", "test_data.pkl"
        )
        train, _ = TrainTestSplit(train).train_test_split(
            tt_ratio, data_path, "train_data.pkl", "validation_data.pkl"
        )
        full_data_sampler = SMOTESampler(self.data_path, merged_data)
        full_data_sampler.SMOTE()
        full_data_sampler.write_output('full_balanced_dataset.pkl')
        fd_scaler = Scaler(self.data_path, 'full_balanced_dataset.pkl')
        fd_scaler.standardize_train_only()
        ohe  = OneHotEncoder(
            self.data_path,
            'full_dataset0.pkl'
        )
        ohe.OHE()
        ohe.write_output('full_balanced_dataset.pkl')
        sampler = SMOTESampler(self.data_path, train)
        sampler.SMOTE()
        sampler.write_output()
        standard_scaler = Scaler(self.data_path)
        standard_scaler.drop_columns()
        standard_scaler.standardize_train_valid()
        standard_scaler.combine_train_valid()
        standard_scaler.standardize_train_test()
        ohe = OneHotEncoder(
            self.data_path
        )
        ohe.OHE()
        ohe.write_output()


if __name__ == "__main__":
    model_instance = WooperModel(data_path)
    if step == 1:
        model_instance.basic_transformations("dataset0.json.gz", "data.info")
    if step == 2:
        model_instance.advanced_transformations()