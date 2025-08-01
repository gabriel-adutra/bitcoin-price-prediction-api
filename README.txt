# Machine Learning Model Integrated via REST API to Predict Bitcoin Price
# How to run the project:

# Open the terminal or command prompt, navigate to the folder containing the files, and run the command below to create a virtual environment:

conda create --name apiBitcoinPrice python=3.12

# Activate the environment:

conda activate apiBitcoinPrice (ou: source activate apiBitcoinPrice)

# Install pip and the dependencies:

conda install pip
pip install -r requirements.txt 

# Open the terminal or command prompt, navigate to the folder where the files are located, and start the API using the command below:

python app.py

# Open another terminal or command prompt, navigate to the folder where the files are located, activate the virtual environment with the dependencies (conda activate apiBitcoinPrice), and run the client app using the command below:

./predict_bitcoin_price.sh

# Use the commands below to deactivate the virtual environment and remove the environment (optional):

conda deactivate
conda remove --name apiBitcoinPrice --all


