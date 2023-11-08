import pandas as pd
from pathlib import Path
import pickle
from sklearn.ensemble import BaggingClassifier


data_dir = Path(__file__).resolve().parents[1]/'data'
random_state = 42
n_jobs = -1

def extract_df(df):
    return df.drop(['label'], axis = 1), df['label']

def get_data(train: str = 'train_OHE.pkl', validate: str = 'validation_OHE.pkl' , trainf: str = 'train_final_OHE.pkl', test:str = 'test_final_OHE.pkl'):
    train = pd.read_pickle(data_dir/train)
    trainx, trainy = extract_df(train)
    validation = pd.read_pickle(data_dir/validate)
    validationx, validationy = extract_df(validation)
    trainf = pd.read_pickle(data_dir/trainf)
    trainfx, trainfy = extract_df(trainf)
    test = pd.read_pickle(data_dir/test)
    testx, testy = extract_df(test)
    return trainx, trainy, validationx, validationy, trainfx, trainfy, testx, testy

if __name__ == '__main__':
    trainx, trainy, validationx, validationy, trainfx, trainfy, testx, testy = get_data()
    bag = BaggingClassifier(n_estimators= 200, n_jobs = n_jobs, random_state = random_state)
    bag.fit(trainfx, trainfy)
    with open(data_dir/'bagged_200trees.pkl', 'wb') as pickle_file:
        pickle.dump(bag, pickle_file)
    