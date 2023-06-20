# import libraries
import os
from typing import Dict, Any
import toml

# path locations
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(PROJECT_PATH, "models")
DATA_PATH = os.path.join(PROJECT_PATH, "data")
TRAIN_FILE_PATH = os.path.join(DATA_PATH, "fraudTrain.csv")
TEST_FILE_PATH = os.path.join(DATA_PATH, "fraudTest.csv")
OUTPUT_PATH = os.path.join(PROJECT_PATH, "output")
PREDICTION_PATH = os.path.join(OUTPUT_PATH, "model prediction")
METRICS_PATH = os.path.join(OUTPUT_PATH, "model performance")

# data related configs
COLS_TO_DROP = ['trans_date_trans_time', 'cc_num', 'first', 'last', 'street', 'zip', 'lat', 'long', 'dob', 'unix_time', 'trans_num', 'merch_lat', 'merch_long'] # indicates which columns to drop
COLS_TO_OH = ['merchant', 'category', 'gender', 'city', 'state', 'job', 'bank_num']
COLS_DATA_DRIFT = ['amt', 'city_pop', 'is_fraud']

DATA_DRIFT_THRESHOLD = 10 # threshold set for data drift
HYPERPARAMETER_TUNING = False

# To read model config file in toml format
def read_model_config() -> Dict[str, Any]:
    filepath = os.path.join(PROJECT_PATH, "model_config.toml")
    with open(filepath, "r", encoding="utf-8") as f:
        return toml.load(f)