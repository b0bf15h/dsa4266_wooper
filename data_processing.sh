python main.py -d ./data -s 1
chmod +x ./data/BioM.R
Rscript ./data/BioM.R
python main.py -d ./data -s 2
cd data
rm *.csv interm.pkl outliers_length.pkl train_data.pkl test_data.pkl validation_data.pkl balanced_train.pkl
cd ..