import pandas as pd
from sklearn.preprocessing import StandardScaler
from data_processing.SMOTE import SMOTESampler
import pickle
import pathlib


class Scaler(object):
    """Standardises data, supports other operations such as combining training and validation data and saving the scaler instance
    As well as scaling the inference data using the saved scaler instance
    """

    def __init__(self, data_path: pathlib.Path, train_file: str = 'balanced_train.pkl', validation_file: str = 'validation_data.pkl', test_file: str = 'test_data.pkl'):
        """Reads in the various data files"""
        self.data_path = data_path
        self.train = pd.read_pickle(self.data_path / train_file)
        self.col_order = self.train.columns
        self.validation = pd.read_pickle(self.data_path / validation_file) 
        self.test = pd.read_pickle(self.data_path / test_file)
        self.new_train = None

    def drop_columns(self):
        """
        Drops categorical columns which are not used for model training or inference
        """
        self.test.drop(
            ["transcript_position", "transcript_length", "transcript_id", "gene_id"],
            axis=1,
            inplace=True,
        )
        self.test = self.test[self.col_order]
        self.validation.drop(
            ["transcript_position", "transcript_length", "transcript_id", "gene_id"],
            axis=1,
            inplace=True,
        )
        self.validation = self.validation[self.col_order]
    def standardize_train_only(self, train_file:str = 'full_dataset0.pkl', scaler_name:str = 'scaler_ds0.pkl'):
        train_copy = self.train.copy(deep=True)
        train_numeric_cols = train_copy.select_dtypes(include=[float])
        if 'relative_sequence_position' in train_numeric_cols.columns:
            train_numeric_cols.drop(["relative_sequence_position"], axis=1, inplace=True)
        scaler = StandardScaler()
        train_copy[train_numeric_cols.columns] = scaler.fit_transform(
            train_numeric_cols
        )
        # save the scaler for transforming other datasets
        # with open(self.data_path/scaler_name, "wb") as pickle_file:
        #     pickle.dump(scaler, pickle_file)
        train_copy.to_pickle(self.data_path / train_file)
    def standardize_train_valid(self, train_file: str = 'train.pkl', validation_file: str = 'validation.pkl'):
        """
        Standardise copies of self.train and self.validation based on distribution of self.train
        Writes output to train.pkl and validation.pkl
        """
        train_copy = self.train.copy(deep=True)
        train_numeric_cols = train_copy.select_dtypes(include=[float])
        train_numeric_cols.drop(["relative_sequence_position"], axis=1, inplace=True)
        scaler = StandardScaler()
        train_copy[train_numeric_cols.columns] = scaler.fit_transform(
            train_numeric_cols
        )
        train_copy.to_pickle(self.data_path / train_file)
        validation_copy = self.validation.copy(deep=True)
        validation_numeric_cols = validation_copy.select_dtypes(include=[float])
        validation_numeric_cols.drop(
            ["relative_sequence_position"], axis=1, inplace=True
        )
        scaler = StandardScaler()
        validation_copy[validation_numeric_cols.columns] = scaler.fit_transform(
            validation_numeric_cols
        )
        validation_copy.to_pickle(self.data_path / validation_file)

    def combine_train_valid(self, scaler_name: str = 'scaler.pkl'):
        """First oversamples validation, then merge with train, and perform standardisation. Also saves the scaler object for future use"""
        smote = SMOTESampler(self.data_path, self.validation)
        balanced_valid = smote.SMOTE()
        balanced_valid = balanced_valid[self.col_order]
        new_train = pd.concat([self.train, balanced_valid])
        train_numeric_cols = new_train.select_dtypes(include=[float])
        train_numeric_cols.drop(["relative_sequence_position"], axis=1, inplace=True)
        scaler = StandardScaler()
        new_train[train_numeric_cols.columns] = scaler.fit_transform(train_numeric_cols)
        # with open(self.data_path / scaler_name, "wb") as pickle_file:
        #     pickle.dump(scaler, pickle_file)
        self.new_train = new_train

    def standardize_train_test(self, scaler_name: str = 'scaler.pkl', train_file: str = 'train_final.pkl', test_file: str = 'test_final.pkl' ):
        """loads specified scaler and scales test data, before writing train_final.pkl and test_final.pkl"""
        train_numeric_cols = self.new_train.select_dtypes(include=[float])
        train_numeric_cols.drop(["relative_sequence_position"], axis=1, inplace=True)
        # with open(self.data_path / scaler_name, "rb") as pickle_file:
        #     scaler = pickle.load(pickle_file)
        scaler = StandardScaler()
        test_copy = self.test.copy(deep=True)
        test_numeric_cols = test_copy.select_dtypes(include=[float])
        test_numeric_cols.drop(["relative_sequence_position"], axis=1, inplace=True)
        test_copy[test_numeric_cols.columns] = scaler.fit_transform(test_numeric_cols)
        test_copy.to_pickle(self.data_path / test_file)
        self.new_train.to_pickle(self.data_path / train_file)