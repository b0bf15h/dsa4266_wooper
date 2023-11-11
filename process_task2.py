from data_processing.DataProcessing import DataParsing, SummariseDataByTranscript, MergeData
from data_processing.InferenceScaler import InferenceProcessor
import pandas as pd
import argparse
import pathlib

parser = argparse.ArgumentParser(prog = 'process_inference_data.py', description = 'Process new batch of data to be ready for inference')
parser.add_argument('--data_path', '-d', type = pathlib.Path, action = 'store' , required = True, help = 'Path to raw data and labels')
parser.add_argument('--step', '-s', type = int, action = 'store', required = True, help = 'Step 1 is parsing json data and creating df, to be done in python. Step 2 requires output from R script which queries Biomart for additional features')
args = parser.parse_args()
data_path = args.data_path
step = args.step

class WooperModel(object):
    def __init__(self, data_path):
        self.raw_data = []
        self.raw_info = []
        self.data_path = data_path
        self.df = None
        # self.reference = []

    # Task 1
    def parse(self, raw_data, index:str = 'data.index'):
        self.raw_data = self.data_path/raw_data
        self.info = self.raw_data/index
        parsed_data = DataParsing(self.raw_data).unlabelled_data(False)
        summarised_data = SummariseDataByTranscript(parsed_data).summarise()
        MergeData(summarised_data, self.info, self.data_path).write_data_for_R()
        # MergeData(parsed_data, self.info, self.data_path).write_data_for_R(df_name = 'reads.pkl')
    def feature_engineer(self, output_filename, csv_data:str = 'biomart_data.csv'):
        csv_data = self.data_path/csv_data
        step1_data = self.data_path/'interm.pkl'
        self.df = pd.read_pickle(step1_data)
        # reads = pd.read_pickle(self.data_path/'reads.pkl')
        # merged_reads = MergeData(reads, csv_data, self.data_path).merge_with_features()
        # merged_reads.to_pickle(self.data_path/(f'reads_{output_filename}'))
        merged_data = MergeData(self.df, csv_data, self.data_path).merge_with_features()
        merged_data.to_pickle(self.data_path/output_filename)
        worker = InferenceProcessor(self.data_path, output_filename)
        worker2 = InferenceProcessor(self.data_path, output_filename)
        worker2.drop_columns()
        worker2.encode()
        worker2.write_output(f'unnormalised_{output_filename}')
        worker.drop_columns()
        worker.scale()
        worker.encode()
        worker.write_output(output_filename)


if __name__ == "__main__":
    model_instance = WooperModel(data_path)
    if step == 1:
        model_instance.parse(
            "MCF7_R3r1",
        )
    if step==2:
        model_instance.feature_engineer(
            "MCF7_R3r1.pkl",
            # "dataset3_tx_length.csv"
        )
        
        
