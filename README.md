# Wooper Model Guide
We recommend using a new instance at least as powerful as t3.2xlarge(8 CPUs, 32GiB RAM) as processing raw data is quite memory intensive.   

The timings of the process are obtained on a new t3.2xlarge instance.
## Python Environment Set-up via Miniconda (~2 mins)
The following steps assume you have not installed Anaconda on your machine. If you already have conda,
simply
```
conda create -n testenv python==3.9.0
```
, git clone the repo and skip to the Python Installations section.    


Within your home directory, run the following command to clone into this repository.    

```
git clone https://github.com/b0bf15h/dsa4266_wooper.git
```
Add the following line to your .bash_profile in home directory, 
since the default setting for .bash_profile is read-only, you need to force-write via :wq! in vim
```
source ~/.bashrc
```
Add the following line to your .bashrc in home directory
```
export PATH='/home/ubuntu/miniconda3/bin:$PATH'
```
Run the conda_setup.sh shell script using the following commands.     

You will be prompted to install new packages, proceed with 'y'
```
cp dsa4266_wooper/conda_setup.sh .
chmod +x conda_setup.sh
./conda_setup.sh
```
Run your .bashrc
```
source ~/.bashrc
```

## Python Installations (~30 s)

Run the following codes to install dependencies. 
```
conda activate testenv
pip install -r dsa4266_wooper/requirements.txt
```

## R Environment Set-up (~13 mins)
Run the r_setup.sh shell script using the following commands.   

You will be prompted to get archives about 1 minute in, proceed with 'Y'
```
cp dsa4266_wooper/r_setup.sh .
chmod +x r_setup.sh
./r_setup.sh
```

## Generating Training Data (~3 mins using dataset0.json.gz and data.info)
Cd into dsa4266_wooper, you can save some time by running the following command and making all shell scripts executable
```
chmod +x *.sh
```
We only support raw data in json format and labels in csv format.
To generate training data, move your zipped raw data(dataset0.json.gz) and labels(data.info) into dsa4266_wooper/data.
Afterwards, from dsa4266_wooper run 
```
./data_processing.sh
```
We use 0.8 for both train-test split and train-validation split. 
You can set your own ratio by modifying the following line in data_processing.sh
```
python main.py -d ./data -s 2  # add -r 0.7 for 70-30 split
```
This script will output 4 pickled dataframes, the first 2 can be used for hyperparameter tuning while the next 2 are used to evaluate model performance.
"train_OHE.pkl" , "validation_OHE.pkl", "train_final_OHE.pkl", "test_final_OHE.pkl".


