from data_processing.DataProcessing import DataParsing, SummariseDataByTranscript, MergeData
from data_processing.InferenceScaler import InferenceProcessor
import pandas as pd
import argparse
import pathlib

parser = argparse.ArgumentParser(prog = 'process_inference_data.py', description = 'Process new batch of data to be ready for inference')
parser.add_argument('--data_path', '-d', type = pathlib.Path, action = 'store' , required = True, help = 'Path to raw data and labels')
parser.add_argument('--step', '-s', type = int, action = 'store', required = True, help = 'Step 1 is parsing json data and creating df, to be done in python. Step 2 requires output from R script which queries Biomart for additional features')
parser.add_argument(
    "--data_name",
    "-dn",
    type=str,
    action="store", 
    required = False,
    help = "Name of dataset to process/predict/train on"
)
parser.add_argument(
    "--info_name",
    "-in",
    type=str,
    action="store", 
    required = False,
    default = 'biomart_data.csv',
    help = "Name of queried csv data "
)
args = parser.parse_args()
data_path = args.data_path
step = args.step
data_name = args.data_name
info_name = args.info_name

class WooperModel(object):
    def __init__(self, data_path):
        self.raw_data = []
        self.raw_info = []
        self.data_path = data_path
        self.df = None
        # self.reference = []

    # Task 1
    def parse(self, raw_data:str):
        self.raw_data = self.data_path/raw_data
        parsed_data = DataParsing(self.raw_data).unlabelled_data(fname = 'data.json', unzip=True)
        summarised_data = SummariseDataByTranscript(parsed_data).summarise()
        MergeData(summarised_data, None, self.data_path).write_data_for_R("unlabelled")
    def feature_engineer(self, output_filename, csv_data):
        csv_data = self.data_path/csv_data
        step1_data = self.data_path/'interm.pkl'
        self.df = pd.read_pickle(step1_data)
        merger = MergeData(self.df, csv_data, self.data_path)
        merged_data = merger.merge_with_features()
        merged_data = merger.drop_unused_features(merged_data)
        merged_data.to_pickle(self.data_path/output_filename)
        worker = InferenceProcessor(self.data_path, output_filename)
        worker.drop_columns()
        worker.scale()
        worker.encode()
        worker.write_output(output_filename)


if __name__ == "__main__":
    model_instance = WooperModel(data_path)
    if step == 1:
        model_instance.parse(
            data_name
        )
    if step==2:
        model_instance.feature_engineer(
            data_name,
            info_name
        )
        
        
