import logging
import pandas as pd
import yaml
import json
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import dill as pickle
from sklearn.base import BaseEstimator, TransformerMixin
# Add the parent directory of utils to the Python path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_loader import load_data
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load configuration
with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)
    
DATA_FILE = config["train_data"]
MODEL_FILE = config["model_file"]
SCORES_FILE = config["scores_file"]
TEST_SIZE = config["test_size"]
RANDOM_STATE = config["random_state"]
FEATURES = config["features"]
TARGET = config["target"]
MODEL_NAME = config["model"]["name"]
MODEL_PARAMS = config["model"]["params"]

# Custom transformer to encode and decode labels
class LabelEncoderDecoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoder = LabelEncoder()

    def fit(self, X, y=None):
        if y is not None:
            self.label_encoder.fit(y)
        return self

    def transform(self, X):
        import pandas as pd
        if isinstance(X, pd.Series):
            return self.label_encoder.transform(X)
        return X

    def inverse_transform(self, X):
        import pandas as pd
        if isinstance(X, pd.Series):
            return self.label_encoder.inverse_transform(X)
        return X

def main():
    logging.info("Loading data...")
    data = load_data(DATA_FILE)
    X = data[FEATURES]
    y = data[TARGET]
    

    #logging.info("Encoding labels...")
    #label_encoder = LabelEncoder()
    #y_encoded = label_encoder.fit_transform(y)

    logging.info("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    logging.info("Training model...")
    if MODEL_NAME == "logistic_regression":
        MODEL = LogisticRegression(**MODEL_PARAMS)
    else:
        raise ValueError(f"{MODEL_NAME} is not supported.")
    
    pipeline = Pipeline([
        ("label_encoder", LabelEncoderDecoder()),
        ("classifier", MODEL)
    ])
    pipeline.fit(X_train, y_train)
    
    logging.info("Testing model...")
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")

    # Save scores to a JSON file
    scores = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }
    with open(SCORES_FILE, "w") as f:
        json.dump(scores, f, indent=4)

    logging.info("Saving model...")
    

    #full_pipeline = FullPipelineWithDecoder(pipeline, label_encoder)

    with open(MODEL_FILE, "wb") as file:
        pickle.dump(pipeline, file)
    logging.info("Model saved successfully.")
    logging.info(f"Model results saved to {SCORES_FILE}")

if __name__ == "__main__":
    main()