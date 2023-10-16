sudo apt update -qq
sudo apt install --no-install-recommends software-properties-common dirmngr
wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | sudo tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
sudo add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"
sudo apt -y install make gcc g++ zlib1g-dev libcurl4-openssl-dev libxml2-dev libssl-dev libpng-dev
libbz2-dev liblzma-dev liblapack-dev libblas-dev gfortran
sudo apt install --no-install-recommends r-base
sudo Rscript ./dsa4266_wooper/installation.R
