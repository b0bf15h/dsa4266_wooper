# Wooper Model

## Installation
Within the ubuntu instance, run the following commands to clone this repo into the ubuntu instance

```
git clone https://github.com/b0bf15h/dsa4266_wooper.git
sudo apt install python3-pip
mkdir data
aws s3 cp --no-sign-request s3:abc ./data
#insert setting up of conda env

cd
pip install -r requirements.txt
```
