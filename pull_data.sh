file_id="1n-wTfvj-lM1jb6l2C-_UtwktvW7yR_cg"
# Replace 'output_filename.extension' with the desired output file name and extension
output_file="model.pkl"

# Download the file using gdown
gdown "https://drive.google.com/uc?id=$file_id" -O "$output_file"