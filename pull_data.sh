mkdir ./data/prediction_data
mkdir ./models
ds0='12FGrDUy6hWm8B3U3nbZxFxIXIAy3XMrZ'
ds0_name='dataset0.json.gz'
ds0_info='10KB36iB1lY_jybFwzrDfQTVI6vrUwpBb'
ds0_infon='data.info'
ds1='1BhgOM9UfJKYyjYSaZxBMOhLpB8SYiP97' 
ds1_name='dataset1.json.gz'
ds2='1CXL5-BE7OlqkjvHvkKoWx40nrQN7yf69'
ds2_name='dataset2.json.gz'
ds3='191dbr5PjbPF-H4ga9geP5AaYaHPc5hUO'
ds3_name='dataset3.json.gz'
ds3_csv='1Xmbt9PZExTFz63DqUDIL8Zk0XLBKHQBe'
ds3_csv_name='dataset3_tx_length.csv'
rf='1yTfHRyuF-wupwZxSlb1XIajMBktRr5SR'
rf_name='rf_final_model.pkl'
rf_nrp='1CvUbJ5YvfezE8Hz-38h-VAiMPjqZv6ZH'
rf_nrp_name='rf_no_rp_final_model.pkl'
gdown "https://drive.google.com/uc?id=$ds0_info" -O "$ds0_infon"
gdown "https://drive.google.com/uc?id=$ds0" -O "$ds0_name"
gdown "https://drive.google.com/uc?id=$ds1" -O "$ds1_name"
gdown "https://drive.google.com/uc?id=$ds2" -O "$ds2_name"
gdown "https://drive.google.com/uc?id=$ds3" -O "$ds3_name"
gdown "https://drive.google.com/uc?id=$ds3_csv" -O "$ds3_csv_name"
gdown "https://drive.google.com/uc?id=$rf" -O "$rf_name"
gdown "https://drive.google.com/uc?id=$rf_nrp" -O "$rf_nrp_name"
mv data.info ./data/
mv dataset0.json.gz ./data/
mv dataset1.json.gz ./data/
mv dataset2.json.gz ./data/
mv dataset3.json.gz ./data/
mv dataset3_tx_length.csv ./data/
mv rf_final_model.pkl ./models
mv rf_no_rp_final_model.pkl ./models