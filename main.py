from data_processing.DataProcessing import DataParsing, SummariseDataByTranscript, MergeData
from data_processing.SMOTE import SMOTESampler
from data_processing.TrainTestSplit import TrainTestSplit
import pandas as pd
import argparse
import pathlib

parser = argparse.ArgumentParser(prog = 'main.py', description = 'End-to-End Model Training')
parser.add_argument('--data_path', '-d', type = pathlib.Path, action = 'store' , required = True, help = 'Path to raw data and labels')
parser.add_argument('--train_test_ratio', '-r', type = float, action = 'store', required = False, help = 'Ratio for train-test split', default = 0.8)
parser.add_argument('--step', '-s', type = int, action = 'store', required = True, help = 'Step 1 is parsing json data and creating df, to be done in python. Step 2 requires output from R script which queries Biomart for additional features')
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
        self.raw_data = self.data_path/raw_data
        self.raw_info = self.data_path/raw_info
        parsed_data = DataParsing(self.raw_data).unlabelled_data()
        summarised_data = SummariseDataByTranscript(parsed_data).summarise()
        MergeData(summarised_data, self.raw_info, self.data_path).write_data_for_R()
    def advanced_transformations(self, csv_data):
        csv_data = self.data_path/csv_data
        step1_data = self.data_path/'interm.pkl'
        self.df = pd.read_pickle(step1_data)
        merged_data = MergeData(self.df, csv_data, self.data_path).merge_with_features()
        TrainTestSplit(merged_data).train_test_split(tt_ratio, data_path)
        sampler = SMOTESampler(self.data_path)
        sampler.SMOTE()
        sampler.write_output()


if __name__ == "__main__":
    model_instance = WooperModel(data_path)
    if step == 1:
        model_instance.basic_transformations(
            "dataset0.json.gz",
            "data.info",
        )
    if step==2:
        model_instance.advanced_transformations(
            "biomart_data.csv"
        )
        
        
