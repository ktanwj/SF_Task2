""" 
!code abstracted from https://github.com/achyutb6/grid-search-cv/blob/master/SciKitLab2.py
I included a portion of the grid search cv code from the above github link and added my own hyperparameters to tune.
-------------------------------------------------------------------------------------------
This script would ingest the processed data and perform grid-search-cv based on the tuned_parameters defined. 

I will be performing hyperparameter tuning on 4 different algorithm:
1. Decision tree
2. KNeighbours
3. XGBoost

The best performing hyperparameters will then be pasted in the model_config.toml for further use.

Note: By running python ./model_tuning.py > ./output/hyperparam_tuning.txt on terminal, the output would be saved under the output folder named: hyperparam_tuning.txt.
Perhaps future iteration could include a method that would do so automatically.
 """

# import libraries
from __future__ import print_function

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# import paths
from config import read_model_config
from utils import DataStore

def model_tuning():
    print(__doc__)
    # load processed data
    datastore = DataStore()

    conf = read_model_config()
    df = datastore.get_processed("df_train_processed.csv")

    # Preprocessing the dataset
    X = df[conf['features']]
    y = df[conf['target']]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split dataset into 20:80 - test and training set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=conf['test_size'], random_state=0)

    # Parameters to be tuned. 
    tuned_parameters = [[{'class_weight': ['balanced'], 'max_depth' : [10,100,1000], 'min_samples_split' : [2,10], 'min_samples_leaf': [1,5], 'max_features' : ["sqrt","log2"]}],
                        # [{'n_neighbors' : [5,10,20],'weights': ['uniform','distance'], 'algorithm' : ['ball_tree', 'kd_tree', 'brute'],'p' : [1,2,3]}],
                        [{'booster': ['gbtree', 'gblinear' ,'dart'], 'learning_rate' : [0.1,0.05,0.2], 'min_child_weight' : [1, 5], 'max_delta_step' : [0, 1]}]
                        ]

    algorithms = [DecisionTreeClassifier(),XGBClassifier()]
    algorithm_names = ["DecisionTreeClassifier","XGBClassifier"]

    # main hyperparameter tuning step
    for i in range(0, len(algorithms)):
        print("################   %s   ################" %algorithm_names[i])
        scores = ['f1']

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            clf = GridSearchCV(algorithms[i], tuned_parameters[i], cv=5,
                            scoring='%s' % score)
            clf.fit(X_train, y_train)

            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                    % (mean, std * 2, params))
            print()

            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            y_true, y_pred = y_test, clf.predict(X_test)
            print(classification_report(y_true, y_pred))
            print("Detailed confusion matrix:")
            print(confusion_matrix(y_true, y_pred))
            print("Precision Score: \n")
            print(precision_score(y_true, y_pred))

            print()


if __name__ == "__main__":
    model_tuning()