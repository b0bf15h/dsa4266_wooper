python process_task2.py -d ./data -s 1
chmod +x ./data/BioM.R
Rscript ./data/BioM.R
python process_task2.py -d ./data -s 2
cd data
rm interm.pkl bmart.csv
cd ..