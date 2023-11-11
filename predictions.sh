mkdir ./data/prediction_data
./pull_data.sh
./process_inference_data.sh 
python main.py -d ./data -s 3 -m ./models -mn 'model.pkl' -dn 'prediction_data.pkl'
cd data
rm biomart_data.csv prediction_data_ids_and_positions.pkl prediction_data.pkl
cd prediction_data
rm data.json
cd ../..