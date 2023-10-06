# Wooper Model

## Environment Set-up
Within the ubuntu instance, run the following commands to clone this repo into the ubuntu instance

```
git clone https://github.com/b0bf15h/dsa4266_wooper.git
```
Add the following line to your .bash_profile
```
source ~/.bashrc
```
Add the following lines to your .bashrc in home directory
```
export PATH=’/home/ubuntu/miniconda3/bin:$PATH’
conda activate testenv # optional, activates the env after running conda_setup.sh and on subsequent terminal launches
```
Run the conda_setup.sh shell script using the following commands 
```
chmod +x conda_setup.sh
./conda_setup.sh
```

## Installation
After activating testenv, install dependencies via the following code.

```
sudo apt install python3-pip
mkdir data
aws s3 cp --no-sign-request s3:abc ./data

cd
pip install -r requirements.txt
```
