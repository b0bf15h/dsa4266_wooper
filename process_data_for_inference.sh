python process_inference_data.py -d ./data -s 1
chmod +x ./data/BioM.R
Rscript ./data/BioM.R
python process_inference_data.py -d ./data -s 2
cd data
rm *.csv interm.pkl outliers_length.pkl
cd ..