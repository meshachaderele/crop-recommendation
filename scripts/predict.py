import dill as pickle
import logging
import yaml
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_loader import load_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load configuration
with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

MODEL_FILE = config["model_file"]
DATA_FILE = config["test_data"]
PREDICTIONS_FILE = config["predictions_file"]


def main():
    with open(MODEL_FILE, 'rb') as file:
        model = pickle.load(file)
        
    data = load_data(DATA_FILE)
    new_data = data.drop(["id", "crop"], axis=1)
    predictions = model.predict(new_data)
    # Decode the prediction
    #if hasattr(model.named_steps['label_encoder'], 'inverse_transform'):
        #predictions = model.named_steps['label_encoder'].inverse_transform([predictions])
    
    data["crop_prediction"] = predictions

    data.to_csv(PREDICTIONS_FILE, index=False)
    logging.info(f"Predictions saved to {PREDICTIONS_FILE}")


if __name__ == "__main__":
    main()
