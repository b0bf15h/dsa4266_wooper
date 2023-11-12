python process_task2.py -d ./data -s 1 -dn 'A549_R5r1'
chmod +x ./data/BioM.R
Rscript ./data/BioM.R
python process_task2.py -d ./data -s 2 -dn 'A549_R5r1.pkl' 
# for dataset3, in addition to changing the first line, you should 
# uncomment the following line to replace the above line
# python process_task2.py -d ./data -s 2 -dn 'dataset3.pkl' -in 'dataset3_tx_length.csv'
cd data
rm interm.pkl bmart.csv
cd ..