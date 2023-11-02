""" ###Python script for it:
import pandas as pd
import pathlib
import os


data_path="."

class TrainTestSplit(object):
    def __init__(self,merged_pkl_data):
      self.merged_pkl_data=merged_pkl_data

    def train_test_split(self,train_size, data_path):
      #grouped=pd.Series(self.merged_pkl_data.sort_values(by="count"))
      #train_gene_id=[]
      train_data=self.merged_pkl_data.sample(n=int(train_size*len(self.merged_pkl_data)),random_state=42)
      test_data=self.merged_pkl_data[self.merged_pkl_data.index.isin(train_data.index)==False]
      print("TRAIN-TEST SPLIT SUCCESSFUL")
      #train_data.drop(['transcript_id', 'gene_id'], axis = 1, inplace=True)
      #test_data.drop(['transcript_id', 'gene_id'], axis = 1, inplace=True)
      
      train_data.to_pickle(os.path.join(data_path, 'train_data.pkl'))
      test_data.to_pickle(os.path.join(data_path, 'test_data.pkl'))
      return (train_data,test_data)


##Oversampler old version 
from pathlib import Path
import pandas as pd
from imblearn.over_sampling import SMOTENC
import os

class OverSampler(object):
        def __init__(self,data_path):
            self.data_path = data_path
            self.data = pd.read_pickle(os.path.join(self.data_path,'train_data.pkl'))
        def sample(self):
            small  = self.data.sample(0.005)
            labels = small['label']
            # labels = self.data['label']
            # self.data = self.data.drop(['label'], axis = 1, inplace=True)
            small.drop(['label'], axis = 1, inplace=True)
            sm = SMOTENC(random_state=42, categorical_features=['sequence', 'm1_seq', 'p1_seq'])
            x,y = sm.fit_resample(small,labels)
            labels = pd.Series(y, name='label')
            self.data = x.join(labels)
        def write_output(self):
            self.data.to_pickle(os.path.join(self.data_path,'oversample_train.pkl'))

'''
        if __name__ == "__main__":
            Oversampler = OverSampler(Path(__file__).resolve().parents[1]/'data')
            Oversampler.sample()
            Oversampler.write_output()
'''   
# data_path = Path(__file__).resolve().parents[1]/'data'
# train = pd.read_pickle(data_path/'train_data.pkl')
# labels = train['label']
# train = train.drop(['transcript_id', 'label', 'gene_id'], axis = 1)
# print('pandas ok')


##Random Sampler
from pathlib import Path
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

class RandomSampler(object):
        def __init__(self,data_path, minor_n, major_n):
            self.data_path = data_path
            self.data = pd.read_pickle(os.path.join(self.data_path,'train_data.pkl'))
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
            self.features.to_pickle(os.path.join(self.data_path,'balanced_train.pkl'))



##From here is the onehot experiment 
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
data=pd.read_pickle('merged_data_v2.pkl')
train_data,test_data=TrainTestSplit(data).train_test_split(0.8,data_path)
#oversampler=OverSampler(data_path)
#Oversampler.sample()
#Oversampler.write_output()
#train_data=pd.read_pickle('oversample_train.pkl')
Oversampler = RandomSampler(data_path,5000,5000)
Oversampler.oversample_minority()
Oversampler.undersample_majority()
Oversampler.write_output()

train_data=pd.read_pickle('balanced_train.pkl')


##To encode "sequence","m1_seq","p1_seq" these three columns
enc=preprocessing.OneHotEncoder(sparse=False,drop='first', handle_unknown='ignore')
enc.fit(train_data.select_dtypes('object')[["sequence","m1_seq","p1_seq"]])
train_transformed_sequences_vars=enc.transform(train_data.select_dtypes('object')[["sequence","m1_seq","p1_seq"]])
train_encoded_frame=pd.DataFrame(train_transformed_sequences_vars)
#encode single digit for -1 and +1, then encode full sequence
new_column_names = {i: f'Encoded_{i}' for i in range(0,63)}
train_encoded_frame.rename(columns=new_column_names, inplace=True)

##set Numeric columns and drop info columns to another dataset.
train_numeric_frame=train_data.drop(["gene_id","transcript_id","transcript_position","sequence","m1_seq","p1_seq"],axis=1)
train_numeric_frame=train_numeric_frame.reset_index()
del train_numeric_frame['index']
train_label_frame=train_data[["gene_id","transcript_id","transcript_position"]]
train_label_frame=train_label_frame.reset_index()
del train_label_frame['index']

##Concat the numeric frame and the encoded frames:
#use it to sample only 10000rows if needed:
import random
random.seed(42)
sampled_indexes=random.sample(list(train_numeric_frame.index),10000)
#For vertical concatenation,use axis=1
encoded_train_data=pd.concat([train_numeric_frame.iloc[sampled_indexes],train_encoded_frame.iloc[sampled_indexes],train_label_frame.iloc[sampled_indexes]],axis=1)
#encoded_train_data


##SVM Model Training
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score,roc_auc_score,classification_report

encoded_train_data['label']=encoded_train_data['label'].astype("int")
encoded_train_data=encoded_train_data
X_train,X_test,y_train,y_test=train_test_split(encoded_train_data.drop("label",axis=1),encoded_train_data["label"],test_size=0.3,random_state=42)

X_train_reference=X_train[["gene_id","transcript_id","transcript_position"]]
X_train=X_train.drop(["gene_id","transcript_id","transcript_position"],axis=1)
X_test_reference=X_test[["gene_id","transcript_id","transcript_position"]]
X_test=X_test.drop(["gene_id","transcript_id","transcript_position"],axis=1)

clf=SVC(kernel='linear')
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

print("Accuracy:",accuracy_score(y_test,y_pred))
print("ROC_AUC_Score:",roc_auc_score(y_test,y_pred))
print("\nF1 Score:",f1_score(y_test,y_pred,average="weighted"))
print("\nClassification Report:\n",classification_report(y_test,y_pred))



print('----------------------------------------------------------------')



#Now try to get the score of each test instance
'''
y_test_score=[]
for i in range(len(y_test)):
     y_pred_score=clf.score(X_test.iloc[i],y_test.iloc[i])
     y_test_score.append(y_pred_score)
X_test_reference['score']=y_test_score.tolist()
X_test_reference
'''
#Naive Bayes result is the no.of 1's/ total test points(3000)

##GBC Model Training
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

scaler=MinMaxScaler()
#X_train=scaler.fit_transform(X_train)
#X_test=scaler.transform(X_test)
'''
parameters = {"n_estimators":[8, 16, 32, 64, 100],
              "learning_rate":[1, 0.5, 0.25],  
              "max_features":[1,5,10,25,50],
              "max_depth":[1,5,10,15,20]}
gb_clf=GradientBoostingClassifier(random_state=42)
GS_clf = GridSearchCV(gb_clf, parameters,cv=5)
GS_clf.fit(X_train,y_train)
print("best params for Boosting:",GS_clf.best_estimator_.get_params())

#gb_clf=GradientBoostingClassifier(n_estimators=100,learning_rate=0.25,max_features=50,max_depth=20,random_state=42)
gb_clf.fit(X_train,y_train)
y_pred=gb_clf.predict(X_test)

print("Accuracy Score for training:",gb_clf.score(X_train,y_train))
print("Classification report for training:")


''' """



import pandas as pd
import numpy as np
import pathlib
import os
data_path="."

import optuna.integration.lightgbm as lgb
from lightgbm import early_stopping
from lightgbm import log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score,roc_auc_score,classification_report

class TrainTestSplit(object):
    def __init__(self,merged_pkl_data):
      self.merged_pkl_data=merged_pkl_data

    def train_test_split(self,train_size, data_path):
      #grouped=pd.Series(self.merged_pkl_data.sort_values(by="count"))
      #train_gene_id=[]
      train_data=self.merged_pkl_data.sample(n=int(train_size*len(self.merged_pkl_data)),random_state=42)
      test_data=self.merged_pkl_data[self.merged_pkl_data.index.isin(train_data.index)==False]
      print("TRAIN-TEST SPLIT SUCCESSFUL")
      #train_data.drop(['transcript_id', 'gene_id'], axis = 1, inplace=True)
      #test_data.drop(['transcript_id', 'gene_id'], axis = 1, inplace=True)
      
      train_data.to_pickle(os.path.join(data_path, 'train_data.pkl'))
      test_data.to_pickle(os.path.join(data_path, 'test_data.pkl'))
      return (train_data,test_data)

#data=pd.read_pickle('balanced_train.pkl')
#TrainTestSplit(data).train_test_split(0.8,data_path)
train=pd.read_pickle('train.pkl')
test=pd.read_pickle('test_final.pkl')
validation=pd.read_pickle('validation.pkl')
train_final=pd.read_pickle('train_final.pkl')
""" train_reference=train[["gene_id","transcript_id","transcript_position"]]
test_reference=test[["gene_id","transcript_id","transcript_position"]]
validation_reference=validation[["gene_id","transcript_id","transcript_position"]]
train=train.drop(["gene_id","transcript_id","transcript_position"],axis=1)
test=test.drop(["gene_id","transcript_id","transcript_position"],axis=1)
validation=validation.drop(["gene_id","transcript_id","transcript_position"],axis=1) """


X_train=train.drop("label",axis=1)
y_train=train["label"]
X_test=test.drop("label",axis=1)
y_test=test["label"]
X_val=validation.drop("label",axis=1)
y_val=validation["label"]
X_train_final=train_final.drop("label",axis=1)
y_train_final=train_final["label"]

categorical_features = ["sequence", "m1_seq", "p1_seq"]
for c in categorical_features:
    X_train[c]=X_train[c].astype("category")
for c in categorical_features:
    X_train_final[c]=X_train_final[c].astype("category")
for c in categorical_features:
    X_test[c]=X_test[c].astype("category")
for c in categorical_features:
    X_val[c]=X_val[c].astype("category")

# Get the column indexes for the specified categorical features
feature_indexes = [X_train_final.columns.get_loc(col) for col in categorical_features]
##use train data and validation data to fine tune the parameters first
dtrain = lgb.Dataset(X_train, label=y_train,categorical_feature=feature_indexes)
dval= lgb.Dataset(X_val, label=y_val,categorical_feature=feature_indexes)
dtrain_final=lgb.Dataset(X_train_final,label=y_train_final,categorical_feature=feature_indexes)
params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
    }

model = lgb.train(
        params,
        dtrain,
        valid_sets=[dtrain, dval],
        callbacks=[early_stopping(100), log_evaluation(100)],
    )

best_params = model.params
print("Best params:", best_params)
#conda activate wp
print("\n--------------------------\nnow train the gbm model with best params:")
import lightgbm as lgb
model=lgb.train(best_params,dtrain_final)
val_preds=model.predict(X_val)
val_preds=np.rint(val_preds)
val_scores=model.predict(X_val,raw_scores=True)

train_final_preds=model.predict(X_train_final)
train_final_preds=np.rint(train_final_preds)
train_final_scores=model.predict(X_train_final,raw_scores=True)
print("training result using the best_params:\n")
print("Accuracy Score for training:",accuracy_score(train_final_preds,y_train_final))
print("ROC_AUC_Score:",roc_auc_score(y_train_final,train_final_scores))
print("\nClassification report for training:",classification_report(train_final_preds,y_train_final))

test_preds=model.predict(X_test)
test_preds=np.rint(test_preds)
test_scores=model.predict(X_test,raw_scores=True)
print("For test results: ")
print("ROC_AUC_Score:",roc_auc_score(y_test,test_scores))
print("\nClassification report for validating:",classification_report(test_preds,y_test))



###Finally save the model: 
print("\n save the model in the end: ")


import pickle

pickle.dump(model, open('model_with_rp.pkl', 'wb'))
#model.save_model('model.pkl')










