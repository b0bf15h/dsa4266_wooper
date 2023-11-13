# Wooper Model Guide
We recommend **launching a new Ubuntu20 04 Large instance on ResearchGateway** at least as powerful as t3.2xlarge(8 CPUs, 32GiB RAM) as processing raw data is quite memory intensive.   

The timings of the process are obtained on a new t3.2xlarge instance.
## Python Environment Set-up via Miniconda (~2 mins)
The following steps assume you have not installed Anaconda on your machine. If you **already have conda**,
simply
```
conda create -n testenv python==3.9.0
```
, git clone the repo and skip to the **Python Installations** section.    \
\
The following steps apply if you launched a fresh instance and do not have conda.  \
\
Within your home directory, run the following command to clone into this repository.    

```
git clone https://github.com/b0bf15h/dsa4266_wooper.git
```
Add the following line to your **.bash_profile** in home directory, since the default setting for .bash_profile is read-only, you need to **force-write via :wq!** in vim
```
source ~/.bashrc
```
Add the following line to your **.bashrc** in home directory
```
export PATH='/home/ubuntu/miniconda3/bin:$PATH'
```
Run the **conda_setup.sh** shell script using the following commands. \
\
You will be prompted to install new packages, proceed with 'y'
```
cp dsa4266_wooper/conda_setup.sh .
chmod +x conda_setup.sh
./conda_setup.sh
```
Run your **.bashrc**
```
source ~/.bashrc
```

## Python Installations (~30 s)

Run the following codes to install dependencies. 
```
conda activate testenv
pip install -r dsa4266_wooper/requirements.txt
# the below line is optional but allows you to try ensembling methods
pip install git+https://github.com/scikit-learn-contrib/DESlib
```

## R Environment Set-up (~13 mins)
Run the **r_setup.sh** shell script using the following commands. \
\
You will be prompted to get archives about 1 minute in, proceed with 'Y'
```
cp dsa4266_wooper/r_setup.sh .
chmod +x r_setup.sh
./r_setup.sh
```

## Generating Training Data (~2.5 mins using dataset0.json.gz and data.info)
Change directory into **dsa4266_wooper**.
```
cd dsa4266_wooper
```
You can save some time by running the following commands and make all shell scripts executable,\
the shell script pulls all needed data for the repository to run **(~30s)**.
```
chmod +x *.sh
./pull_data.sh
```
We only support **raw data in json format and labels in csv format**.\
\
Afterwards, from dsa4266_wooper run 
```
./data_processing.sh
```
The relative path to the raw data and labels from **dsa4266_wooper/data** should be specified using the flags -dn and -ln respectively
\
For example, this is the correct command in the shell script if you are using **dataset0.json.gz and data.info**.
```
python main.py -d ./data -s 1 -dn 'dataset0.json.gz' -ln 'data.info'
```

We use 0.8 for both train-test split and train-validation split. \
You can set your own ratio by modifying the following line in data_processing.sh
```
python main.py -d ./data -s 2  # add -r 0.7 for 70-30 split
```
This script will output 5 pickled dataframes into **dsa4266_wooper/data**, train and validation are used for hyperparameter tuning, train_final and test_final are used to evaluate model performance, full_balanced_dataset can be used to train the final model since it contains the most information. \
\
**E.g.** "train_OHE.pkl" , "validation_OHE.pkl", "train_final_OHE.pkl", "test_final_OHE.pkl", "full_balanced_dataset.pkl"

### If your raw data is not zipped, then modify the unlabelled_data() function in the respective python scripts run by the shell script
**E.g.** Replacing the first line with the second line
```
parsed_data = DataParsing(self.raw_data).unlabelled_data()

parsed_data = DataParsing(self.raw_data).unlabelled_data(fname = 'data.json', unzip = False)

```
The **fname** argument is only valid if **unzip = False** and should refer to the .json file containing read-level data.

### Some features may be null
As we query for data from Ensembl using Biomart, some versioning issues with transcript IDs and the database will result in failed queries. \
\
Queried features include relative_sequence_position for inference data, and 9 other features for analysis data. \
\
This affects synthetic datasets such as dataset2.json.gz, as well as all 12 of the SGNex datasets, between 50 and 900 transcripts will have queried features be null for the 12 datasets.

## Preparing Raw Data for Inference (~2 mins using dataset3.json.gz)
To prepare unlabelled raw data for inference, run the following command from the **dsa4266_wooper** directory
```
./process_inference_data.sh
```
The relative path to the raw data from **dsa4266_wooper/data** and the name of the output file should be specified using the -dn flags in the shell script when calling the python script. \
\
For example, for dataset1, the following lines will do.
```
python process_inference_data.py -d ./data -s 1 -dn 'dataset1.json.gz'
python process_inference_data.py -d ./data -s 2 -dn 'dataset1.pkl'
```
For dataset3.json.gz you should **uncomment the specified line described in the shell script**. \
\
This script will output 2 pickled dataframes and 1 csv file into **dsa4266_wooper/data**, one pickle file is used as input to the model while the other one is used as an index since it contains identifying information for each sequence. \
\
The csv file contains queried data from Ensembl database. This is not present for dataset2 as it is fully synthetic as well as dataset3 as plant sequencing data is not found on Ensembl \
\
**E.g.** dataset1.pkl, dataset1_ids_and_positions.pkl and biomart_data.csv

## Preparing Raw Data for Further Analysis (~2.5 mins using SGNex_A549_directRNA_replicate5_run1)
To prepare unlabelled raw data for further analysis, run the following command from the **dsa4266_wooper** directory
```
./process_task2.sh
```
The relative path to the raw data from **dsa4266_wooper/data** and the name of the output file should be specified using the -dn flags in the shell script when calling the python script. \
\
For example, for the provided data A549_Replicate5_Run1, the following lines will do.
```
python process_task2.py -d ./data -s 1 -dn 'A549_R5r1'
python process_task2.py -d ./data -s 2 -dn 'A549_R5r1.pkl' 
```
This script will output 3 pickled dataframes and 1 csv file into **dsa4266_wooper/data**, similar to the outputs of **process_inference_data.sh**, the extra pickle file contains unnormalised data which may be useful for analysis.\
\
**E.g.** A549_R5r1.pkl, unnormalised_A549_R5r1.pkl, A549_R5r1_ids_and_positions.pkl and biomart_data.csv

## Training Model (~4 mins)
To train both our models (with and without relative sequence position), run the following command from the **dsa4266_wooper** directory
```
./train_model.sh
```
This script will output the 2 models into **dsa4266_wooper/models** 

## Making Predictions (~4.5 mins using datasets 1,2,3)
To make predictions on the provided test data, run the following command from the **dsa4266_wooper** directory
```
./predictions.sh
```
This script processes raw data for inference, and makes prediction on it using the correct model.\
Random Forest with relative sequence position for datasets 1 and 3, Random Forest without relative sequence position for dataset2.
\
The predictions will be written to **dsa4266_wooper/data/prediction_data** with the name **dataset1_probs.csv**, **dataset2_probs.csv** **dataset3_probs.csv**  .
