from sklearn import preprocessing
import pandas as pd
import pickle
from pathlib import Path

def one_hot_encode(df, encoder_available=False):
    file_name = 'encoder.pkl'
    if not encoder_available:
        enc = preprocessing.OneHotEncoder(sparse=False,drop='first', handle_unknown='ignore')
        encoded_data = enc.fit_transform(df[["sequence", "m1_seq", "p1_seq"]])
        with open(data_path/file_name, 'wb') as pickle_file:
            pickle.dump(enc, pickle_file)
    else:
        with open(data_path/file_name, 'rb') as pickle_file:
            enc = pickle.load(pickle_file)
        encoded_data = enc.transform(df[["sequence", "m1_seq", "p1_seq"]])
    encoded_column_names = enc.get_feature_names_out(["sequence", "m1_seq", "p1_seq"])
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_column_names)
    df = df.reset_index(drop=True)
    result_df = pd.concat([df, encoded_df], axis=1)
    result_df.drop(columns=["sequence", "m1_seq", "p1_seq"], inplace=True)
    return result_df


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parents[1]/'data'
    train_final = pd.read_pickle(data_path/'train_final.pkl')
    test_final = pd.read_pickle(data_path/'test_final.pkl')
    train = pd.read_pickle(data_path/'train.pkl')
    validation = pd.read_pickle(data_path/'validation.pkl')
    one_hot_encode(train_final).to_pickle(data_path/'train_final_OHE.pkl')
    one_hot_encode(test_final, True).to_pickle(data_path/'test_final_OHE.pkl')
    one_hot_encode(train, True).to_pickle(data_path/'train_OHE.pkl')
    one_hot_encode(validation, True).to_pickle(data_path/'validation_OHE.pkl')