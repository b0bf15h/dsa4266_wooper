from pathlib import Path
import pandas as pd
from imblearn.over_sampling import SMOTENC

class OverSampler(object):
        def __init__(self,data_path):
            self.data_path = data_path
            self.data = pd.read_pickle(self.data_path/'train_data.pkl')
        def sample(self):
            small  = self.data.sample(0.1)
            labels = small['label']
            # labels = self.data['label']
            # self.data = self.data.drop(['label'], axis = 1, inplace=True)
            small.drop(['label'], axis = 1, inplace=True)
            sm = SMOTENC(random_state=42, categorical_features=['sequence', 'm1_seq', 'p1_seq'])
            x,y = sm.fit_resample(small,labels)
            labels = pd.Series(y, name='label')
            self.data = x.join(labels)
        def write_output(self):
            self.data.to_pickle(self.data_path/'oversample_train.pkl')

if __name__ == "__main__":
    Oversampler = OverSampler(Path(__file__).resolve().parents[1]/'data')
    Oversampler.sample()
    Oversampler.write_output()
    
# data_path = Path(__file__).resolve().parents[1]/'data'
# train = pd.read_pickle(data_path/'train_data.pkl')
# labels = train['label']
# train = train.drop(['transcript_id', 'label', 'gene_id'], axis = 1)
# print('pandas ok')
# sm = SMOTENC(random_state=42, categorical_features=['transcript_position', 'sequence', 'm1_seq', 'p1_seq'])
# print('initiate smotenc ok')
# x,y = sm.fit_resample(train,labels)