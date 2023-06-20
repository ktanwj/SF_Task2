# import ML libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# import helper functions
from utils import read_train_data, read_test_data, preprocess_data, detect_data_drift
from config import HYPERPARAMETER_TUNING, COLS_DATA_DRIFT
from model import train_evaluate_models, check_model_drifts
from model_tuning import model_tuning

# ----------------------------------------------------------------------- #
# main script - run this
# ----------------------------------------------------------------------- #
def main():

    # reads csv file stored locally
    print('------------- Read/pull train and test data -------------')
    df_train = read_train_data()
    df_test = read_test_data()

    # check for data shift vs. current latest dataset
    print('------------- Checking for data drifts -------------')
    for col in COLS_DATA_DRIFT:
        if detect_data_drift(df_train, df_train, col=col, alpha=0.05):
            print(f'Attention: data drift found in {col}. Please check data')
        else:
            print(f'no data drift found in {col}.')

    # apply data pre-processing techniques to check, clean and apply feature engineering to the data
    print('------------- Processing data -------------')
    print('Processing Training dataset ...')
    df_train = preprocess_data(df_train, file_name="df_train_processed")

    print('Processing Test dataset ...')
    df_test = preprocess_data(df_test, file_name="df_test_processed")
    
    # # re-run model_tuning.py (hyperparam tuning) if HYPERPARAMETER_TUNING == True
    if HYPERPARAMETER_TUNING:
        print('------------- Performing Hyperparameter tuning -------------')
        model_tuning()
        print('------------- Hyperparameter tuning complete -------------')

    # # train and evaluate models saved in model_config
    mapping = {'best_xgb': XGBClassifier}

    print('------------- Start train/evaluate models -------------')
    train_evaluate_models(df_train, df_test, mapping)

    # check for model drifts and flag out if there is a change (by comparing results with base DecisionTreeClassifier)
    print('------------- Checking for model drifts -------------')
    check_model_drifts(mapping, reference_model_name='default_dt')

if __name__ == "__main__":
    main()