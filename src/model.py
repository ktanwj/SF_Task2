# import libraries
from abc import ABC, abstractmethod
from importlib.util import module_for_loader
import json
from random import Random
from typing import Dict, List
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# import paths
from config import read_model_config, OUTPUT_PATH
from utils import DataStore

# ML libraries
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report

# ----------------------------------------------------------------------- #
# Base sklearn classifier
# ----------------------------------------------------------------------- #
"""
Abstract base class for defining Classifiers with three methods:
1. train
2. evaluate
3. predict
"""
class Classifier(ABC):
    @abstractmethod
    def train(self, *params) -> None:
        pass

    @abstractmethod
    def evaluate(self, *params) -> Dict[str, float]:
        pass

    @abstractmethod
    def predict(self, *params) -> np.ndarray:
        pass

"""
SklearnClassifier defines a base estimator that will be layered on to models such as decision trees.
This defines the required steps that would be performed for each model, as well as align the metrics that would be used to evaluate different models.
"""
class SklearnClassifier(Classifier):
    def __init__(
        self, estimator: BaseEstimator, features: List[str], target: str,
    ):
        self.clf = estimator
        self.features = features
        self.target = target
        self.datastore = DataStore()

    def train(self, df_train: pd.DataFrame):
        # train model based on features selected in model_config
        self.clf.fit(df_train[self.features].values, df_train[self.target].values)

    def evaluate(self, df_test: pd.DataFrame, pred_path:str):
        # make prediction using model
        y_pred = self.clf.predict(df_test[self.features].values)
        y_true = df_test[self.target].values
        y_pred_proba = self.predict(df_test)

        # model scores
        accuracy = accuracy_score(y_true, y_pred)
        roc = roc_auc_score(y_true, y_pred_proba)
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

        # save predictions 
        self.datastore.put_predictions(pred_path, pd.DataFrame(y_pred))

        return {'accuracy':accuracy, 'roc':roc,
                'f1': f1, 'precision':precision, 'recall':recall}

    def predict(self, df: pd.DataFrame):
        return self.clf.predict_proba(df[self.features].values)[:, 1]

# ----------------------------------------------------------------------- #
# Base scripts for modelling
# ----------------------------------------------------------------------- #
def train_evaluate_model(df_train, df_test, model_name, user_classifier):
    """ To train and evaluate model

     Parameters
    ----------
    df_train : pd.DataFrame
        training data in dataframe

    df_test : pd.DataFrame
        test data in dataframe
    
    model_name : str
        name of model -- this has to correspond to the model name defined in model_config.toml

    user_classifier : Classifier
        type of classifier e.g. DecisionTreeClassifier

    Returns
    -------
    None. Trains and evaluate model, and stores model outputs into respective folders.
     """
    #  initialisation
    datastore = DataStore()
    config = read_model_config()
    np.random.seed(config['seed_num'])

    # define classifier and load config based on model name
    classifier = user_classifier(**config[model_name])
    model = SklearnClassifier(classifier, config['features'], config['target'])

    # train and evaluate
    print(f'training model {model_name} ...')
    model.train(df_train)

    print(f'evaluate model {model_name} ...')
    metrics = model.evaluate(df_test, pred_path = model_name + '_pred.csv')

    # store variables
    datastore.put_model(model_name + ".pkl", model)
    datastore.put_metrics(model_name + "_metrics.json", metrics)

def generate_report(df_test:pd.DataFrame, target:str, model_name:str):
    """ (unused) generate reports such as confusion and classification """
    datastore = DataStore()

    y_true = df_test[target].values
    y_pred = datastore.get_predictions(model_name + '_pred.csv')

    class_report = classification_report(y_true, y_pred)
    confusion = confusion_matrix(y_true, y_pred)

    # generate confusion plot
    conf_mat = pd.DataFrame(confusion_matrix(y_true, y_pred))  
    fig = plt.figure(figsize=(10, 7))  
    sns.heatmap(conf_mat, annot=True, annot_kws={"size": 16}, fmt="g")  
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")  
    plt.ylabel("True Label")  
    plt.savefig(f'{model_name}_confusion_mat.png')

# ----------------------------------------------------------------------- #
# Main model scripts
# ----------------------------------------------------------------------- #
def train_evaluate_models(df_train: pd.DataFrame, df_test: pd.DataFrame, mapping:json) -> None:
    """
    Evaluates best model using the best model config output from hyperparameter tuning.
    Parameters
    ----------
    df : pd.DataFrame
        processed dataframe

    mapping : json
        contains mapping of model names : model classifier

    Returns
    -------
    None. Trains and evaluate model, and stores model outputs into respective folders.
    """
    config = read_model_config()
    model_names = config['best_model_configs']

    for model_name in model_names:
        classifier = mapping[model_name]
        train_evaluate_model(df_train, df_test, model_name, classifier)
    print('------------- Model run complete -------------')

def check_model_drifts(mapping:json, reference_model_name:str)->bool:
    """
    Evaluates best model using the best model config output from hyperparameter tuning.
    Parameters
    ----------
    mapping : json
        contains mapping of model names : model classifier

    Returns
    -------
    Boolean. True if there is model drift, false otherwise.
    """
    f_ref = open(f"{OUTPUT_PATH}/model performance/{reference_model_name}_metrics.json")
    ref_data = json.load(f_ref)
    for target_model_name in mapping:
        f_target = open(f"{OUTPUT_PATH}/model performance/{target_model_name}_metrics.json")
        target_data = json.load(f_target)
        for metric in ['accuracy', 'roc','f1','precision']:
            if target_data[metric] < ref_data[metric]:
                print(f'model drift detected as {metric} for target model is lower than baseline model')
                return True
            else:
                print(f'No model drift detected in {metric}')
    return False

if __name__ == "__main__":
    # run to test model.py
    datastore = DataStore()
    config = read_model_config()
    model_name = 'default_dt'
    classifier = DecisionTreeClassifier
    df = datastore.get_processed('df_processed.csv') # read processed df from datastore
    df_train, df_test = train_test_split(df, test_size=config["test_size"])
    assert df.shape[0] == df_train.shape[0] + df_test.shape[0]
    train_evaluate_model(df_train=df_train, df_test=df_test, model_name = model_name, user_classifier=classifier)
    generate_report(df_test, config['target'], model_name = model_name)
