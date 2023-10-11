from pathlib import Path
import pandas as pd

class DownSampler(object):
        def __init__(self,data_path):
            self.data_path = data_path
            self.train = pd.read_pickle(self.data_path/'train_data.pkl')
            self.test = pd.read_pickle(self.data_path/'test_data.pkl')
        def sample_data_n(self, sample_size):
            train_n = sample_size[0]
            test_n = sample_size[1]
            train_sampled  = self.train.sample(n = train_n, random_state = 42)
            test_sampled = self.test.sample(n = test_n, random_state = 42)
            self.train = train_sampled
            self.test = test_sampled
            return None    
        def sample_data_p (self, sample_size):
            train_p = sample_size[0]
            test_p = sample_size[1]
            train_sampled  = self.train.sample(p = train_p, random_state = 42)
            test_sampled = self.test.sample(p = test_p, random_state = 42)
            self.train = train_sampled
            self.test = test_sampled
            return None        
        def sample_data(self, mode, sample_size):
            '''
            Samples data independently from train and test data, using number of samples or proportion'
            '''
            if (type(mode)!=str):
                print('Mode must be a string')
            if (mode == 'n'):
                self.sample_data_n(sample_size)
                return
            self.sample_data_p(sample_size)
        def write_output(self):
            self.train.to_pickle(self.data_path/'downsample_train.pkl')
            self.test.to_pickle(self.data_path/'downsample_test.pkl')
            print(f"Training data has {len(self.train)} rows")
            print(f"Test data has {len(self.test)} rows")

if __name__ == "__main__":
    Downsampler = DownSampler(Path(__file__).resolve().parents[1]/'data')
    Downsampler.sample_data('n',[8000,2000])
    Downsampler.write_output()