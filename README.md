# Wooper Model
We recommend using a new instance at least as powerful as t3.2xlarge(8 CPUs, 32GiB RAM) as processing raw data is quite memory intensive.   

The timings of the process are obtained on a new t3.2xlarge instance.
## Python Environment Set-up via Miniconda (~1 min)
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
Add the following line to your .bash_profile in home directory
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

You will be prompted to get archives less than 2 mins in, proceed with 'Y'
```
cp dsa4266_wooper/r_setup.sh .
chmod +x r_setup.sh
./r_setup.sh
```


