# Wooper Model

## Environment Set-up
The following steps assume you have not installed Anaconda on your machine. If you already have conda,
simply
```
conda create -n testenv python==3.9.0
```
, git clone the repo and skip to the Installation section.

Within your home directory, run the following command to clone into this repository.

```
git clone https://github.com/b0bf15h/dsa4266_wooper.git
```
Add the following line to your .bash_profile in home directory
```
source ~/.bashrc
```
Add the following line to your .bashrc in home directory
```
export PATH='/home/ubuntu/miniconda3/bin:$PATH'
```
Run the conda_setup.sh shell script using the following commands 
```
cp dsa4266_wooper/conda_setup.sh .
chmod +x conda_setup.sh
./conda_setup.sh
```
Run your .bashrc
```
source ~/.bashrc
```

## Installation

Run the following codes to install dependencies.
```
conda activate testenv
pip install -r dsa4266_wooper/requirements.txt
mkdir data
aws s3 cp --no-sign-request s3:abc ./data
```
