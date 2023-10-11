from pathlib import Path
import pandas as pd

class DownSampler(object):
        def __init__(self,data_path):
            self.data_path = data_path
            self.train = pd.read_pickle(self.data_path/'train_data.pkl')
            self.test = pd.read_pickle(self.data_path/'test_data.pkl')
            self.train_sampled = []
            self.test_sampled = []
        def sample_data_n(self, sample_size):
            train_n = sample_size[0]
            test_n = sample_size[1]
            self.train_sampled.append(self.train.sample(n = train_n, random_state = 42))
            self.test_sampled.append(self.test.sample(n = test_n, random_state = 42))
            l = len(self.test_sampled)-1
            self.train =  pd.concat([self.train, self.train_sampled[l], self.train_sampled[l]]).drop_duplicates(keep=False)
            self.test =  pd.concat([self.test, self.test_sampled[l], self.test_sampled[l]]).drop_duplicates(keep=False)
        def sample_data_p (self, sample_size):
            train_p = sample_size[0]
            test_p = sample_size[1]
            self.train_sampled.append(self.train.sample(p = train_p, random_state = 42))
            self.test_sampled.append(self.test.sample(p = test_p, random_state = 42))
            l = len(self.test_sampled)-1
            self.train =  pd.concat([self.train, self.train_sampled[l], self.train_sampled[l]]).drop_duplicates(keep=False)
            self.test =  pd.concat([self.test, self.test_sampled[l], self.test_sampled[l]]).drop_duplicates(keep=False)    
        def sample_data(self, mode, sample_size, num_samples):
            '''
            Samples data independently from train and test data, using number of samples or proportion'
            '''
            if (type(mode)!=str):
                print('Mode must be a string')
            if (mode == 'n'):
                for i in range(num_samples):
                    self.sample_data_n(sample_size)
                    print(f"Training data has {len(self.train)} rows")
                    print(f"Testing data has {len(self.test)} rows")   
                return
            for i in range(num_samples):
                self.sample_data_p(sample_size)
                print(f"Training data has {len(self.train)} rows")
                print(f"Testing data has {len(self.test)} rows")   
        def write_output(self):
            for i in range(len(self.train_sampled)):
                train_name = 'downsample_train_'+str(i+1)+'.pkl'
                test_name = 'downsample_test'+str(i+1)+'.pkl'
                self.train_sampled[i].to_pickle(self.data_path/train_name)
                self.test_sampled[i].to_pickle(self.data_path/test_name)

if __name__ == "__main__":
    Downsampler = DownSampler(Path(__file__).resolve().parents[1]/'data')
    Downsampler.sample_data('n',[8000,2000],3)
    Downsampler.write_output()