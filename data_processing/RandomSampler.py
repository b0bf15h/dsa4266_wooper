from pathlib import Path
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

class RandomSampler(object):
        def __init__(self,data_path, minor_n, major_n):
            self.data_path = data_path
            self.data = pd.read_pickle(self.data_path/'train_data.pkl')
            self.labels = self.data['label']
            self.features = self.data.drop(['label'], axis = 1)
            self.minor_n = minor_n
            self.major_n = major_n
            self.major_initial_n = self.labels.value_counts().max()
            self.minor_initial_n = self.labels.value_counts().min()
        def oversample_minority(self):
            assert self.minor_initial_n < self.minor_n, "Oversampling requires supplied value to be greater than the current count"
            ros = RandomOverSampler(sampling_strategy= {0:self.major_initial_n,1:self.minor_n}, random_state= 42)
            self.features,self.labels = ros.fit_resample(self.features,self.labels)
        def undersample_majority (self):
            assert self.major_initial_n > self.major_n, "Undersampling requires supplied value to be smaller than the current count"
            rus = RandomUnderSampler(sampling_strategy= {0:self.major_n,1:self.minor_n}, random_state= 42)
            self.features,self.labels = rus.fit_resample(self.features, self.labels)
        def write_output(self):
            '''
            assumes that minority class still has less samples after resampling
            '''
            assert self.labels.value_counts().min() == self.minor_n, "Something is wrong"
            self.features['label'] = self.labels
            self.features.to_pickle(self.data_path/'balanced_train.pkl')

if __name__ == "__main__":
    Oversampler = RandomSampler(Path(__file__).resolve().parents[1]/'data',1500000,1500000)
    Oversampler.oversample_minority()
    Oversampler.undersample_majority()
    Oversampler.write_output()
    