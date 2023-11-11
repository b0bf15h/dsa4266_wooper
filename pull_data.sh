model_id="1k_ql8Hz49Qx75i-ILk4Zua4dPCRNqKE1"
model_file="model.pkl"

data_id="1cyIojmXhihI6OdaPRIqzhpA6ail2J7zj"
data_file="prediction_data.json"

encoder_id="1Bk_XHuD_o-AZY_iPc9cb_0noBMnpfuIo"
encoder_file="encoder_ds0.pkl"

# Download the file using gdown
gdown "https://drive.google.com/uc?id=$model_id" -O "$model_file"
gdown "https://drive.google.com/uc?id=$data_id" -O "$data_file"
gdown "https://drive.google.com/uc?id=$encoder_id" -O "$encoder_file"

mv model.pkl ./models/model.pkl
mv prediction_data.json ./data/prediction_data/data.json
mv encoder_ds0.pkl ./data/encoder_ds0.pkl