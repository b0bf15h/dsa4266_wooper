###Python script for it:
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



#Now try to get the score of each test instance
'''
y_test_score=[]
for i in range(len(y_test)):
     y_pred_score=clf.score(X_test.iloc[i],y_test.iloc[i])
     y_test_score.append(y_pred_score)
X_test_reference['score']=y_test_score.tolist()
X_test_reference
'''
print("\n below is the result using only the difference of -1 and +1 instead of directly input them all:\n")
#Naive Bayes result is the no.of 1's/ total test points(3000)

#####
#####
###Now try to modify the m1 and p1 seqs.  
train_data["m1_seq"]=train_data["m1_seq"].str[0]
train_data["p1_seq"]=train_data["p1_seq"].str[4]
##To encode "sequence","m1_seq","p1_seq" these three columns
enc=preprocessing.OneHotEncoder(sparse=False,drop='first', handle_unknown='ignore')
enc.fit(train_data.select_dtypes('object')[["sequence","m1_seq","p1_seq"]])
train_transformed_sequences_vars=enc.transform(train_data.select_dtypes('object')[["sequence","m1_seq","p1_seq"]])
train_encoded_frame=pd.DataFrame(train_transformed_sequences_vars)
#encode single digit for -1 and +1, then encode full sequence
new_column_names = {i: f'Encoded_{i}' for i in range(0,train_encoded_frame.shape[1])}
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



