python main.py -d ./data -s 1
chmod +x ./data/BioM.R
Rscript ./data/BioM.R
python main.py -d ./data -s 2
cd data
rm bmart.csv train_data.pkl test_data.pkl validation_data.pkl balanced_train.pkl train.pkl validation.pkl train_final.pkl test_final.pkl interm.pkl OHE.pkl biomart_data.csv
cd ..
