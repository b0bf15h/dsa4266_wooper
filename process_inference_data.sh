python process_inference_data.py -d ./data -s 1 -dn 'dataset1.json.gz'
chmod +x ./data/BioM.R
Rscript ./data/BioM.R
python process_inference_data.py -d ./data -s 2 -dn 'dataset1.pkl'
# for dataset3, uncomment the following line to replace the above line
# python process_inference_data.py -d ./data -s 2 -dn 'dataset3.pkl' -in 'dataset3_tx_length.csv'
cd data
rm interm.pkl bmart.csv
cd ..