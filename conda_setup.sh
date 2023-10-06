wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b
export PATH='/home/ubuntu/miniconda3/bin:$PATH'
conda init bash
conda create -n testenv python==3.9.0
