mkdir ./models
full_model_id="1U73TiZqM-txbnu6SQkEr1uBENBGKsrZK"
full_model_file="model.pkl"
gdown "https://drive.google.com/uc?id=$full_model_id" -O "$full_model_file"
mv model.pkl ./models/rf_best_trials_final_tuning.pkl
model_no_rp_id="1U0GanWT9ag5M6XKLIqDZQF3z1Em-8ZMY"
gdown "https://drive.google.com/uc?id=$model_no_rp_id" -O "$full_model_file"
mv model.pkl ./models/rf_best_trials_no_rel_pos.pkl
f_dataset='1CQ_O8n4QXJryVOhDhTqHFPRYS2QEt4cH'
gdown "https://drive.google.com/uc?id=$f_dataset" -O "$full_model_file"
mv model.pkl ./data/full_balanced_dataset.pkl

python main.py -d ./data -s 3 -m ./models -pn 'rf_best_trials_final_tuning.pkl' -dn 'full_balanced_dataset.pkl' -mn 'rf_final_model.pkl'
python main.py -d ./data -s 3 -m ./models -pn 'rf_best_trials_no_rel_pos.pkl' -dn 'full_balanced_dataset.pkl' -mn 'rf_no_rp_final_model.pkl' -rp False