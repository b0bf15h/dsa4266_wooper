from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from pathlib import Path

def random_forest(train):
    X_train = train.drop(columns=['label'])
    y_train = train['label']
    
    # Random Forest with hyper parameter tuning
    rf1 = RandomForestClassifier(n_estimators = 368, max_features='sqrt', random_state = 42, n_jobs=-1, max_depth=11, verbose = 0)
    rf1.fit(X_train, y_train)
    return rf1
    
if __name__ == "__main__":
    data_path = Path(__file__).resolve().parents[1]/'data'
    train = pd.read_pickle(data_path/'train_OHE.pkl')
    random_forest(train)