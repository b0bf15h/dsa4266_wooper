python main.py -d ./data -s 1 -dn 'dataset0.json.gz' -ln 'data.info'
chmod +x ./data/BioM.R
Rscript ./data/BioM.R
python main.py -d ./data -s 2 -r 0.8
cd data
rm bmart.csv train_data.pkl test_data.pkl validation_data.pkl balanced_train.pkl train.pkl validation.pkl train_final.pkl test_final.pkl interm.pkl biomart_data.csv full_dataset0.pkl
cd ..
