python main.py -d ./data -s 1
chmod +x ./data/BioM.R
Rscript ./data/BioM.R
python main.py -d ./data -s 2
python ./data_processing/OHE.py
rm ./data/*.csv
rm ./data/interm.pkl
rm ./data/outliers_length.pkl