from pathlib import Path
import pandas as pd
from imblearn.over_sampling import SMOTENC

class SMOTESampler(object):
        def __init__(self,data_path, data):
            self.data_path = data_path
            self.data = pd.read_pickle(self.data_path/data)
            self.labels = None
            self.features = None
        def SMOTE(self):
            truncated = self.data.drop(['transcript_position', 'transcript_length', 'transcript_id', 'gene_id'], axis = 1)
            self.labels = truncated['label']
            self.features = truncated.drop(['label'],axis = 1)
            sm = SMOTENC(random_state=42, categorical_features=['sequence', 'p1_seq', 'm1_seq'])
            self.features, self.labels = sm.fit_resample(self.features, self.labels)
            self.features['label'] = self.labels
            return self.features
        def write_output(self):
            self.features['label'] = self.labels
            self.features.to_pickle(self.data_path/'balanced_train.pkl')
            print("length of balanced training data: ")
            print(len(self.features))


if __name__ == "__main__":
    Oversampler = SMOTESampler(Path(__file__).resolve().parents[1]/'data')
    Oversampler.SMOTE()
    Oversampler.write_output()
    