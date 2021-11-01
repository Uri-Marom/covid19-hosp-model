import sys
import argparse
from experiment_manager.Experiment import Experiment
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn import svm
from xgboost import XGBClassifier
import json
from sklearn.impute import SimpleImputer

from imblearn.ensemble import BalancedRandomForestClassifier

def parse_params(params_json = None):
    if params_json is None:
        return {}
    with open(params_json) as json_file:
        params = json.load(json_file, object_hook=lambda d: {int(k) if k.lstrip('-').isdigit() else k: v for k, v in d.items()})
        return params


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name_experiment", type=str, help="The name of the experiment")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    name_experiment = args.name_experiment
    csv_target_variable_path = "datasets/v9.csv"
    csv_features_path = "datasets/v9.csv"
    models_hub = {
                    'LogisticRegression': LogisticRegression(),
                    'LinearSVC': svm.LinearSVC(),
                    'KNeighborsClassifier': KNeighborsClassifier(),
                    'RandomForestClassifier': RandomForestClassifier(),
                    'GradientBoostingClassifier': GradientBoostingClassifier(),
                    'LinearRegression': LinearRegression(),
                    'DecisionTreeRegressor': DecisionTreeRegressor(),
                    'RandomForestRegressor': RandomForestRegressor(),
                    'KNeighborsRegressor': KNeighborsRegressor(),
                    'XGBClassifier': XGBClassifier(),
                    'BalancedRandomForestClassifier': BalancedRandomForestClassifier()
    }

    imputers_hub = {
                        'SimpleImputer': SimpleImputer()
    }
    uri = 'file:\\\mkm-cifs\\HOSP_Share\\Experiments'
    experiment = Experiment(experiment_name=name_experiment, features_csv_path=csv_features_path, uri=uri)

    stand = StandardScaler()
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    metrics = [
        # "balanced_accuracy",
        # "f1",
        "roc_auc",
        "precision",
        "recall",
        "average_precision"
    ]

    models = [
        (models_hub['XGBClassifier'], parse_params('models/DefaultXGBoost.json')),
        (models_hub['BalancedRandomForestClassifier'], parse_params('models/DefaultBalancedRandomForest.json'))
        #(models_hub['RandomForestClassifier'], parse_params('models/WeightedRandomForestClassifier.json'))
        # (models_hub['GradientBoostingClassifier'], parse_params('models/DefaultGradientBoostingClassifier.json'))
    ]
    imputers = [
        (imputers_hub['SimpleImputer'], {'strategy': 'mean'}),
        (imputers_hub['SimpleImputer'], {'strategy': 'median'}),
        (imputers_hub['SimpleImputer'], {'strategy': 'most_frequent'})
    ]
    for imputer, imputer_params in imputers:
        for model, model_params_dict in models:
            run = experiment.create_run(model, model_params_dict, cross_validation_method=kf)
            run.add_pre_process_action(imputer.__class__.__name__, imputer, imputer_params)
            run.add_pre_process_action('standard scaler', stand, {})
            run.run()
    print("Finished Experiment")




