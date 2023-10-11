from data_processing.DataProcessing import DataParsing, MergeData
from TrainTestSplit import TrainTestSplit
import argparse
import pathlib

parser = argparse.ArgumentParser(prog = 'main.py', description = 'End-to-End Model Training')
parser.add_argument('--data_path', '-d', type = pathlib.Path, action = 'store' , required = True, help = 'Path to raw data and labels')
parser.add_argument('--train_test_ratio', '-r', type = float, action = 'store', required = False, help = 'Ratio for train-test split', default = 0.8)
args = parser.parse_args()
data_path = args.data_path
tt_ratio = args.train_test_ratio

class WooperModel(object):
    def __init__(self):
        self.raw_data = []
        self.raw_info = []

    # Task 1
    def train_model(self, raw_data, raw_info):
        self.raw_data = raw_data
        self.raw_info = raw_info
        parsed_data = DataParsing(self.raw_data).unlabelled_data()
        merged_data = MergeData(parsed_data, self.raw_info).merge()
        train, test = TrainTestSplit(merged_data).train_test_split(tt_ratio, data_path)
        print("length of training data: ")
        print(len(train))
        print("length of test data: ")
        print(len(test))


if __name__ == "__main__":
    model_instance = WooperModel()
    model_instance.train_model(
        data_path / "dataset0.json.gz",
        data_path / "data.info",
    )
