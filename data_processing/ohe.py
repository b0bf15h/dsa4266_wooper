from sklearn import preprocessing

def one_hot_encode(df):
    enc = preprocessing.OneHotEncoder(sparse=False,drop='first', handle_unknown='ignore')
    encoded_data = enc.fit_transform(df[["sequence", "m1_seq", "p1_seq"]])
    encoded_column_names = enc.get_feature_names_out(["sequence", "m1_seq", "p1_seq"])
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_column_names)
    result_df = pd.concat([df, encoded_df], axis=1)
    result_df.drop(columns=["sequence", "m1_seq", "p1_seq"], inplace=True)
    return result_df