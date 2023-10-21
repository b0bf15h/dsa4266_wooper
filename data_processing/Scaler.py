import pandas as pd
from sklearn.preprocessing import StandardScaler
from data_processing.SMOTE import SMOTESampler
import pickle

class Scaler(object):
    def __init__(self,data_path):
        self.data_path = data_path
        self.train = pd.read_pickle(self.data_path/'balanced_train.pkl')
        self.col_order = self.train.columns
        self.validation = pd.read_pickle(self.data_path/'validation_data.pkl')
        self.test = pd.read_pickle(self.data_path/'test_data.pkl')
        self.new_train = None
    def drop_columns(self):
        self.test.drop(['transcript_position', 'transcript_length', 'transcript_id', 'gene_id'], axis = 1,inplace=True)
        self.test = self.test[self.col_order]
        self.validation.drop(['transcript_position', 'transcript_length', 'transcript_id', 'gene_id'], axis = 1,inplace=True)
        self.validation = self.validation[self.col_order]
    def standardize_train_valid(self):
        train_copy = self.train.copy(deep=True)
        train_numeric_cols = train_copy.select_dtypes(include=[float])
        train_numeric_cols.drop(['relative_sequence_position'], axis = 1, inplace = True)
        scaler = StandardScaler()
        train_copy[train_numeric_cols.columns] = scaler.fit_transform(train_numeric_cols)
        train_copy.to_pickle(self.data_path/'train.pkl')
        validation_copy = self.validation.copy(deep=True)
        validation_numeric_cols = validation_copy.select_dtypes(include=[float])
        validation_numeric_cols.drop(['relative_sequence_position'], axis = 1, inplace = True)
        validation_copy[validation_numeric_cols.columns] = scaler.transform(validation_numeric_cols)
        validation_copy.to_pickle(self.data_path/'validation.pkl')
    def combine_train_valid(self):
        smote = SMOTESampler(self.data_path,'validation_data.pkl')
        balanced_valid = smote.SMOTE() 
        balanced_valid = balanced_valid[self.col_order]
        new_train = pd.concat([self.train, balanced_valid])
        train_numeric_cols = new_train.select_dtypes(include=[float])
        train_numeric_cols.drop(['relative_sequence_position'], axis = 1, inplace = True)     
        scaler = StandardScaler()
        new_train[train_numeric_cols.columns] = scaler.fit_transform(train_numeric_cols)
        file_name = 'scaler.pkl'
        with open(self.data_path/file_name, 'wb') as pickle_file:
            pickle.dump(scaler, pickle_file)
        self.new_train = new_train
    def standardize_train_test(self):
        train_numeric_cols = self.new_train.select_dtypes(include=[float])
        train_numeric_cols.drop(['relative_sequence_position'], axis = 1, inplace = True)
        file_name = 'scaler.pkl'
        with open(self.data_path/file_name, 'rb') as pickle_file:
            scaler = pickle.load(pickle_file)
        test_copy = self.test.copy(deep=True)
        test_numeric_cols = test_copy.select_dtypes(include=[float])
        test_numeric_cols.drop(['relative_sequence_position'], axis = 1, inplace = True)
        test_copy[test_numeric_cols.columns] = scaler.transform(test_numeric_cols)
        test_copy.to_pickle(self.data_path/'test_final.pkl')
        self.new_train.to_pickle(self.data_path/'train_final.pkl')
    def sanity_check(self):
        print(len(self.train))
        print(len(self.validation))
        print(len(self.new_train))
# path = Path(__file__).resolve().parents[1]/'data'
# train = pd.read_pickle(path/'balanced_train.pkl')
# col_order = train.columns

# test = pd.read_pickle(path/'test_data.pkl')
# test.drop(['transcript_position', 'transcript_length', 'transcript_id', 'gene_id'], axis = 1,inplace=True)
# test = test[col_order]

# validation = pd.read_pickle(path/'validation_data.pkl')
# validation.drop(['transcript_position', 'transcript_length', 'transcript_id', 'gene_id'], axis = 1,inplace=True)
# validation = validation[col_order]

# train_numeric_cols = train.select_dtypes(include=[float])
# train_numeric_cols.drop(['relative_sequence_position'], axis = 1, inplace = True)
# scaler = StandardScaler()
# train[train_numeric_cols.columns] = scaler.fit_transform(train_numeric_cols)

# validation_numeric_cols = validation.select_dtypes(include=[float])
# validation_numeric_cols.drop(['relative_sequence_position'], axis = 1, inplace = True)
# validation[validation_numeric_cols.columns] = scaler.transform(validation_numeric_cols)

# test_numeric_cols = test.select_dtypes(include=[float])
# test_numeric_cols.drop(['relative_sequence_position'], axis = 1 , inplace=True)
# test[test_numeric_cols.columns] = scaler.transform(test_numeric_cols)

# train.to_pickle(path/'train_data_final.pkl')
# validation.to_pickle(path/'validation_data_final.pkl')
# test.to_pickle(path/'test_data_final.pkl')

# print('Standard scaling completed')
# print('Saving a scaler for other datasets')


