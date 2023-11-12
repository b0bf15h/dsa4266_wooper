python process_task2.py -d ./data -s 1 -dn 'A549_R5r1'
chmod +x ./data/BioM.R
Rscript ./data/BioM.R
python process_task2.py -d ./data -s 2 -dn 'A549_R5r1.pkl' 
cd data
rm interm.pkl bmart.csv
cd ..