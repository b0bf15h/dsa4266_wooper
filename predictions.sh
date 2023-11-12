python process_inference_data.py -d ./data -s 1 -dn 'dataset1.json.gz'
chmod +x ./data/BioM.R
Rscript ./data/BioM.R
python process_inference_data.py -d ./data -s 2 -dn 'dataset1.pkl'
cd data
rm interm.pkl bmart.csv
cd ..
python process_inference_data.py -d ./data -s 1 -dn 'dataset2.json.gz'
chmod +x ./data/BioM.R
Rscript ./data/BioM.R
python process_inference_data.py -d ./data -s 2 -dn 'dataset2.pkl'
cd data
rm interm.pkl bmart.csv
cd ..
python process_inference_data.py -d ./data -s 1 -dn 'dataset3.json.gz'
chmod +x ./data/BioM.R
Rscript ./data/BioM.R
python process_inference_data.py -d ./data -s 2 -dn 'dataset3.pkl' -in 'dataset3_tx_length.csv'
cd data
rm interm.pkl bmart.csv
cd ..
mv ./data/dataset1.pkl ./data/prediction_data/dataset1.pkl
mv ./data/dataset1_ids_and_positions.pkl ./data/prediction_data/dataset1_ids_and_positions.pkl
python main.py -d ./data/prediction_data -s 4 -m ./models -dn 'dataset1.pkl'
mv ./data/dataset2.pkl ./data/prediction_data/dataset2.pkl
mv ./data/dataset2_ids_and_positions.pkl ./data/prediction_data/dataset2_ids_and_positions.pkl
python main.py -d ./data/prediction_data -s 4 -m ./models -dn 'dataset2.pkl'
mv ./data/dataset3.pkl ./data/prediction_data/dataset3.pkl
mv ./data/dataset3_ids_and_positions.pkl ./data/prediction_data/dataset3_ids_and_positions.pkl
python main.py -d ./data/prediction_data -s 4 -m ./models -dn 'dataset3.pkl'