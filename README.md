# SkillsFuture Case Study 2
This project is done as part of the case study for SkillsFuture.  

**Name: Kelvin Tan Wei Jie**  
**Email: kelvintanweijie@gmail.com**  

## Folder structure
src  
--> models (contains pkl from models trained in this project)  
--> output  
----> hyperparam_tuning.txt (hyperparameter results)  
----> model performance (contains json files documenting model performance)  
----> model prediction (contains csv files documenting predictions for each model)  
--> config.py (configs used for defining PATHS and data-related configs)  
--> main.py (main python script to run)  
--> model.py (contains ML model-related classes/functions)  
--> model_config.toml (configs used to define model-related parameters)  
--> model_tuning.py (contains code to run hyperparameter tuning)  
--> utils.py (contains utility codes and helpful functions for managing files)  
.gitignore (to prevent src/data folder from getting uploaded into Git)  
README.md (readme file for the project)  
eda.ipynb (jupyter lab/notebook containing codes for task 1)  
requirements.txt (packages used for this project)  
run.sh (bash script to run other related scripts in this project)  

## Instructions

Before starting, add in fraudTrain and fraudTest data into the data folder (excluded due to space issues). For executing the pipeline, go to the base folder and type `bash run.sh` in the terminal.

To modify model parameters, go to src/model_config.toml.
Here are a list of configurable parameters:  
1. **features**: defines the independent features used in the model. To change, simply add/drop the feature names onto the list.
2. **target**: defines the target to be trained/evaluated on. For our project, we will be predicting is_fraud hence that is selected.
3. **test_size**: defines the proportion of data to be split into the test set. 
4. **seed_num**: pre-defines the seed number that would be used in model training, for reproducible results.
5. **best_model_configs**: the name of the model configs with the best performance based on the GridSearch CV Hyperparameter tuning.
6. **[model_configs]**: contains the name of the model, as well as the hyperparameters used. Simply change the values to tune the model manually.

To modify the data/path parameters, go to src/config.py.
Here are a list of configurables:
1. **PROJECT_PATH** and others: contains defined paths to the respective folders.
2. **COLS_TO_DROP**: defines the columns that we would like to drop as part of pre-processing data.
4. **COLS_TO_OH**: defines the columns that we would like to perform one-hot-encoding.
5. **HYPERPARAMETER_TUNING**: True if we would like to perform hyperparameter tuning.


## Choice of models

I used two models:
1. Decision Tree (baseline model)  
**Reasons:**
-  the ease of interpretation, as trees with splitting criteria was helpful to understand the problem space
- Secondly, it is non-parametric. This means Decision Trees would likely work well with most data types and distributions since it doesn't make any assumption about the underlying distribution of the data
- Lastly, it is able to handle both categorical and numerical data

2. XGBoost  
**Reasons:**
- less likely to overfit due to its ensemble nature
- It has the best model performance, due to the gradient boosting algorithm which makes weak predictors stronger and the use of a regularization term to further reduce overfitting on training data

## Evaluation of models developed
XGBoost is the overall best performing model across all metrics such as Accuracy, F1 score and ROC.

| Model  (best settings for F1)          | Accuracy |F1 score | ROC |
| -------------                          |:-------------:|:-------------:|:-------------:|
| Decision Tree   | 0.997 | 0.60 | 0.86 |
| XGBoost  | 0.998 | 0.71 | 0.994 |