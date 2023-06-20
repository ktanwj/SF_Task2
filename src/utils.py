# ----------------------------------------------------------------------- #
# Import required modules
# ----------------------------------------------------------------------- #

# import libraries
import os
import numpy as np
import pandas as pd
import functools
import pickle
import json


from typing import Any, Dict
from abc import ABC, abstractmethod
from scipy.stats import entropy
from scipy import stats

# preprocessing libraries
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

pd.options.mode.chained_assignment = None  # default='warn' - to deal with SettingWithCopyWarning in Pandas

# required variables from config file
from config import DATA_PATH, PROJECT_PATH, OUTPUT_PATH, TRAIN_FILE_PATH, TEST_FILE_PATH, MODEL_PATH, PREDICTION_PATH, METRICS_PATH
from config import COLS_TO_DROP, COLS_TO_OH

# ----------------------------------------------------------------------- #
# File read methods
# ----------------------------------------------------------------------- #

def read_train_data(file_path:str = TRAIN_FILE_PATH, test_connection:bool = True) -> pd.DataFrame:
    """ reads csv file from the local path provided
    Parameters
    ----------
    file_path (optional) : str
        Path to the csv file stored locally
    test_connection (optional) : bool
        True if we would like to test the validity of the dataset
        
    Returns
    -------
    pd.DataFrame
        A pandas dataframe containing the returned file
    """
    df = pd.read_csv(file_path, index_col=[0])
    if test_connection:
        if df.empty:
            print('Imported file is empty. Please check.')
        else:
            print('Read train file success!')
    return df

def read_test_data(file_path:str = TEST_FILE_PATH, test_connection:bool = True) -> pd.DataFrame:
    """ reads csv file from the local path provided
    Parameters
    ----------
    file_path (optional) : str
        Path to the csv file stored locally
    test_connection (optional) : bool
        True if we would like to test the validity of the dataset
        
    Returns
    -------
    pd.DataFrame
        A pandas dataframe containing the returned file
    """
    df = pd.read_csv(file_path, index_col=[0])
    if test_connection:
        if df.empty:
            print('Imported file is empty. Please check.')
        else:
            print('Read test file success!')
    return df
    
# ----------------------------------------------------------------------- #
# Data processing methods
# ----------------------------------------------------------------------- #
def preprocess_data(df:pd.DataFrame, file_name:str) -> pd.DataFrame:
    """ main pre-processing step
    Steps applied:
    1. check and remove duplicates (if any)
    2. check for missing values and flag out error (if any)
    3. apply feature engineering to add in features (more details documented in apply_FeatureEngineering)
    4. drop unused columns based on the config file
    5. apply one-hot encoding to categorical variables
    6. stores processed data into folder

    Parameters
    ----------
    df : pd.DataFrame
        input dataframe

    file_name : str
        input name of processed file
        
    Returns
    -------
    df : pd.DataFrame
        Processsed dataframe
    """    
    # step 1: check and remove duplicates (if any)
    if ~check_no_dupes(df):
        df = remove_dupes(df) # remove duplicates    

    # step 2: check for missing values and flag out error (if any)
    check_no_nulls(df) # flags out if there is any missing values in the dataset
    
    # step 3: apply feature engineering to add in features (more details documented in apply_FeatureEngineering)
    df = apply_FeatureEngineering(df)

    # step 4: drop unused columns based on the config file
    df = drop_cols(df, COLS_TO_DROP)

    # step 5: apply one-hot encoding to categorical variables
    df = apply_LabelEncoding(df, COLS_TO_OH)

    # print('------------- Processing complete -------------')

    # # step 6: store processed data into folder
    datastore = DataStore()
    datastore.put_processed(file_name + ".csv", df)
    return df

def check_no_dupes(df:pd.DataFrame) -> bool:
    """ To check pandas DataFrame for duplicated rows
    Parameters
    ----------
    df : pd.DataFrame
        input dataframe
        
    Returns
    -------
    bool : True if no duplicates found, else False
    """
    check = df.isnull().sum().sort_values(ascending=False).sum() == 0
    if check:
        print('No duplicates found. Check passed!')
        return True
    else:
        print('Duplicates found. Check failed, please check!')
        return False
    
def check_no_nulls(df:pd.DataFrame) -> bool:
    """ To check pandas DataFrame for null values
    Parameters
    ----------
    df : pd.DataFrame
        input dataframe
        
    Returns
    -------
    bool : True if no null values found, else False
    """
    check = df.isnull().sum().sort_values(ascending=False).sum() == 0
    if check:
        print('No missing values found. Check passed!')
        return True
    else:
        print(f'{df.isnull().sum().sort_values(ascending=False).sum()} null values found. Check failed!')
        return False

def remove_dupes(df:pd.DataFrame) -> pd.DataFrame:
    """ To remove duplicated rows from input DataFrame. Prints message if verbose is set to true.
    Parameters
    ----------
    df : pd.DataFrame
        input dataframe
    -------
    df : pd.DataFrame
        cleaned dataframe without duplicates
    """
    return df[~df.duplicated()]
    

def drop_cols(df:pd.DataFrame, cols_to_drop:list()) -> pd.DataFrame:
    """  To drop specified columns from DataFrame
    Parameters
    ----------
    df : pd.DataFrame
        input dataframe
    cols_to_drop : list(str)
        indicate columns to drop
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with dropped columns
    """
    return df.drop(columns=cols_to_drop, axis=1)


def convert_str_to_bool(df:pd.DataFrame, cols_to_convert:list()) -> pd.DataFrame:
    """ To convert string columns to boolean type
    Parameters
    ----------
    df : pd.DataFrame
        input dataframe
    
    cols_to_convert : list(str)
        columns to convert from string to boolean

    Returns
    -------
    df : pd.DataFrame
        Dataframe with boolean types as specified
    """
    for col in cols_to_convert:
        df[col] = [1 if r == 'Yes' else 0 for r in df[col]]
    return df

def apply_FeatureEngineering(df:pd.DataFrame) -> pd.DataFrame:
    """ To include new features that would be helpful to model training
    Features include:
    1. trans_month: month based on trans_date_trans_time column, with the numbers representing the month
    2. trans_week: day of week based on trans_date_trans_time column, with 0 as Monday and 6 as Sunday
    3. mmi_num: Major Industry Identifier (MII), represented by the first digit
    4. bank_num: Issuing bank number, represented by the next 5 digits
    5. age_cardholder: age of card holder today based on their date of birth

    Parameters
    ----------
    df : pd.DataFrame
        input dataframe
        
    Returns
    -------
    df : pd.DataFrame
        Dataframe with new features as specified
     """
    df['date'] = pd.to_datetime(df['trans_date_trans_time'])
    df['trans_week'] = [d.weekday() for d in df['date']]
    df['trans_month'] = [int(d.strftime("%m")) for d in df['date']]
    df['mmi_num'] = [str(cc_num)[0] for cc_num in df['cc_num']]
    df['bank_num'] = [str(cc_num)[1:6] for cc_num in df['cc_num']]
    df['age_cardholder'] = [2023 - int(dob[:4]) for dob in df['dob']] # assuming it is currently the year of 2023
    
    return df

def apply_LabelEncoding(df:pd.DataFrame, cols_to_oh:list()) -> pd.DataFrame:
    """ To apply label encoding to the columns in cols_to_oh
    parameters
    ----------
    df : pd.DataFrame
        input dataframe

    cols_to_oh : list(str)
        columns to apply one-hot encoding to
        
    Returns
    -------
    df : pd.DataFrame
        Dataframe with new label encoding features added
    """
    # apply label encoding
    le = LabelEncoder()
    df[cols_to_oh] = df[cols_to_oh].apply(le.fit_transform)
    
    return df

# ----------------------------------------------------------------------- #
# Check data / model drift methods
# ----------------------------------------------------------------------- #

""" code abstracted from: https://towardsdatascience.com/calculating-data-drift-in-machine-learning-53676ff5646b """
def data_length_normalizer(gt_data, obs_data, bins = 100):
    """
    Data length normalizer will normalize a set of data points if they
    are not the same length.
    
    params:
        gt_data (List) : The list of values associated with the training data
        obs_data (List) : The list of values associated with the observations
        bins (Int) : The number of bins you want to use for the distributions
        
    returns:
        The ground truth and observation data in the same length.
    """

    if len(gt_data) == len(obs_data):
        return gt_data, obs_data 

    # scale bins accordingly to data size
    if (len(gt_data) > 20*bins) and (len(obs_data) > 20*bins):
        bins = 10*bins 

    # convert into frequency based distributions
    gt_hist = plt.hist(gt_data, bins = bins)[0]
    obs_hist = plt.hist(obs_data, bins = bins)[0]
    plt.close()  # prevents plot from showing
    return gt_hist, obs_hist 

def softmax(vec):
    """
    This function will calculate the softmax of an array, essentially it will
    convert an array of values into an array of probabilities.
    
    params:
        vec (List) : A list of values you want to calculate the softmax for
        
    returns:
        A list of probabilities associated with the input vector
    """
    return(np.exp(vec)/np.exp(vec).sum())

def calc_cross_entropy(p, q):
    """
    This function will calculate the cross entropy for a pair of 
    distributions.
    
    params:
        p (List) : A discrete distribution of values
        q (List) : Sequence against which the relative entropy is computed.
        
    returns:
        The calculated entropy
    """
    return entropy(p,q)
    
def calc_drift(gt_data, obs_data, col):
    """
    This function will calculate the drift of two distributions given
    the drift type identifeid by the user.
    
    params:
        gt_data (DataFrame) : The dataset which holds the training information
        obs_data (DataFrame) : The dataset which holds the observed information
        col (String) : The data column you want to compare
        
    returns:
        A drift score
    """

    gt_data = gt_data[col].values
    obs_data = obs_data[col].values

    # makes sure the data is same size
    gt_data, obs_data = data_length_normalizer(
        gt_data = gt_data,
        obs_data = obs_data
    )

    # convert to probabilities
    gt_data = softmax(gt_data)
    obs_data = softmax(obs_data)

    # run drift scores
    drift_score = calc_cross_entropy(gt_data, obs_data)
    return drift_score

""" end of code abstracted from: https://towardsdatascience.com/calculating-data-drift-in-machine-learning-53676ff5646b """

def detect_data_drift(reference_df:pd.DataFrame, production_df:pd.DataFrame, col:str, alpha:float = 0.05)->bool: 
    """ To apply ks_test to compare distribution of numeric cols in reference and production df
    parameters
    ----------
    reference_df : pd.DataFrame
        new data to be compared with existing data

    production_df : pd.DataFrame
        existing data

    col : str
        numeric column for comparison
        
    Returns
    -------
    bool : pd.DataFrame
        True if there is data drift, false otherwise
    """
    test = stats.ks_2samp(reference_df[col], production_df[col], alternative = 'two-sided')
    if test[1] < alpha: # reject null hypothesis that F(x) = G(x), hence there is data drift
        return True
    
    return False
    


# ----------------------------------------------------------------------- #
# Store data/model paths and helper methods
# ----------------------------------------------------------------------- #
class InvalidExtension(Exception):
    pass

class Classifier(ABC):
    """
    Base estimator class built using Abstract Base Classes.
    Three methods defined (we can potentially add on more methods in the future if needed):
    1. train: trains ML model 
    2. evaluate: evaluate performance of ML model using the test dataset
    3. predict: performs model prediction
    """
    @abstractmethod
    def train(self, *params) -> None:
        pass

    @abstractmethod
    def evaluate(self, *params) -> Dict[str, float]:
        pass

    @abstractmethod
    def predict(self, *params) -> np.ndarray:
        pass

def _check_filepath(ext):
    """ Wrapper function that helps check validity of filepath """
    def _decorator(f):
        @functools.wraps(f)
        def _wrapper(*args, **kwargs):
            filepath = kwargs.get("filepath")
            if not filepath:
                filepath = args[1]

            if not filepath.endswith(ext):
                raise InvalidExtension(f"{filepath} has invalid extension, want {ext}")

            return f(*args, **kwargs)

        return _wrapper

    return _decorator

class Path:
    """
    Helper function that helps get and pull files from respective folders.
    There are two types of methods defined here:
    1. get_(file_type): returns Dataframe/model.pkl based on the path defined
    2. put_(file_type): stores file on the path defined
    """
    project_path = PROJECT_PATH
    processed_data_path = DATA_PATH
    model_path = MODEL_PATH
    output_path = OUTPUT_PATH
    prediction_path = PREDICTION_PATH
    metrics_path = METRICS_PATH

    @_check_filepath(".csv")
    def get_csv(self, filepath: str, **kwargs) -> pd.DataFrame:
        return pd.read_csv(filepath, **kwargs)

    @_check_filepath(".csv")
    def put_csv(self, filepath: str, df: pd.DataFrame, **kwargs) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df must be of type pd.DataFrame, got {type(df)}")
        df.to_csv(filepath, index=False, **kwargs)

    @_check_filepath(".pkl")
    def get_pkl(self, filepath: str) -> Any:
        with open(filepath, "rb") as f:
            return pickle.load(f)

    @_check_filepath(".pkl")
    def put_pkl(self, filepath: str, python_object: Any) -> None:
        if not python_object:
            raise TypeError("python_object must be non-zero, non-empty, and not None")
        with open(filepath, "wb") as f:
            pickle.dump(python_object, f)

    @_check_filepath(".json")
    def get_json(self, filepath: str) -> Dict:
        with open(filepath, "r") as f:
            return json.load(f)

    @_check_filepath(".json")
    def put_json(self, filepath: str, dic: Dict) -> None:
        if not isinstance(dic, dict):
            raise TypeError(f"dic must be of type dict, got {type(dic)}")
        with open(filepath, "w") as f:
            json.dump(dic, f)
    
class DataStore(Path):
    """ Using path base class, DataStore has a list of helper function that conveniently stores/returns data """
    @_check_filepath(".csv")
    def get_processed(self, filepath: str, **kwargs) -> pd.DataFrame:
        filepath = os.path.join(self.processed_data_path, filepath)
        return self.get_csv(filepath, **kwargs)

    @_check_filepath(".csv")
    def put_processed(self, filepath: str, df: pd.DataFrame, **kwargs) -> None:
        filepath = os.path.join(self.processed_data_path, filepath)
        self.put_csv(filepath, df, **kwargs)

    def get_model(self, filepath: str) -> Classifier:
        filepath = os.path.join(self.model_path, filepath)
        return self.get_pkl(filepath)

    def put_model(self, filepath: str, model: Classifier) -> None:
        filepath = os.path.join(self.model_path, filepath)
        self.put_pkl(filepath, model)

    def get_metrics(self, filepath: str) -> Dict[str, float]:
        filepath = os.path.join(self.metrics_path, filepath)
        return self.get_json(filepath)

    def put_metrics(self, filepath: str, metrics: Dict[str, float]) -> None:
        filepath = os.path.join(self.metrics_path, filepath)
        self.put_json(filepath, metrics)

    def get_predictions(self, filepath: str, **kwargs) -> pd.DataFrame:
        filepath = os.path.join(self.prediction_path, filepath)
        return self.get_csv(filepath, **kwargs)

    def put_predictions(self, filepath: str, df: pd.DataFrame, **kwargs) -> None:
        filepath = os.path.join(self.prediction_path, filepath)
        self.put_csv(filepath, df, **kwargs)
