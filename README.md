# Wooper Model

## Environment Set-up
Within your home directory, run the following commands to clone set-up your environment.

```
git clone https://github.com/b0bf15h/dsa4266_wooper.git
cp dsa4266_wooper/conda_setup.sh .
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
