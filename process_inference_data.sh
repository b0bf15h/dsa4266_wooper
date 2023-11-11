python process_inference_data.py -d ./data -s 1
chmod +x ./data/BioM.R
Rscript ./data/BioM.R
python process_inference_data.py -d ./data -s 2
cd data
rm interm.pkl bmart.csv
cd ..