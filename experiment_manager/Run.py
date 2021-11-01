from pathlib import Path
import mlflow
import mlflow.sklearn
from numpy.distutils.system_info import numarray_info
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve, precision_score, recall_score,\
    average_precision_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
# import eli5


def rmdir(dir_to_delete):
    for child in dir_to_delete.glob('*'):
        if child.is_file():
            child.unlink()
        else:
            rmdir(child)
    dir_to_delete.rmdir()


class Run:
    """Class that represents a configuration for training a model

    Attributes:
        run_name (str): The name of the run
        X (pandas data frame): The data to train on the model
        y (pandas data frame): The target variable
        model (sklearn model object): The model to train
        cv (object): The object that represents the kfold for the cross-validation
        pre_process (list of pipeline methods): The list of all the pre-processing
                                                methods to do before the training of the model
        MID_RESULTS_DIR (str): The name of the directory of mid artifacts

    """
    def __init__(self, run_name, input_features, y, model, model_params_dict, cross_validation_method):
        """

        :param run_name: The name of the run
        :param input_features: The features for training the model
        :param y: The target variable
        :param model: The model object
        :param model_params_dict: The dictionary of the parameters of the model
        :param cross_validation_method: The method to do the cross validation phase
        """
        self.run_name = run_name
        self.X = input_features
        self.y = y
        self.cv = cross_validation_method
        self.MID_RESULTS_DIR = Path('mid_results')
        self.__log_num_splits(self.cv)
        mlflow.log_params(model_params_dict)
        model.set_params(**model_params_dict)
        self.model = model
        self.pre_process = []

    @staticmethod
    def __log_num_splits(cv):
        """

        :param cv: The cross validation object for the cross-validation
        :return: No return value, the function logs the number of splits for the cross-validation
        """
        n_splits = cv.n_splits
        mlflow.log_param('Num Splits', (cv.__class__.__name__, n_splits))

    def add_pre_process_action(self, action_name, action_function, action_params_dict):
        """

        :param action_name: The name of the pre-processing
        :param action_function: The function of the pre-processing
        :param action_params_dict: The dictionary of parameters for the pre-processing
        :return: No return value, the function appends the pre-processing phase and logs it's parameters
        """

        if not action_params_dict:
            mlflow.log_param(action_name, 'No Parameters')
        else:
            mlflow.log_param(action_name, action_params_dict)
            action_function.set_params(**action_params_dict)
        self.pre_process.append((action_name, action_function))

    def __log_roc(self, score_folds):
        """

        :param score_folds: (n_samples, 2) array that holds the probabilities to each sample
        :return: No return value, the method logs the auc of the roc and graph of the roc
        """
        fig, ax = plt.subplots()
        aucs = []
        tprs = []
        mean_fpr = np.linspace(0, 1, 100)
        for i in range(self.cv.n_splits):
            curr_ground_truth_indices, curr_probs = score_folds[i]
            curr_ground_truth = self.y.iloc[curr_ground_truth_indices]
            curr_score = curr_probs[:, 1]
            fpr, tpr, _ = roc_curve(curr_ground_truth, curr_score)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            tprs.append(interp_tpr)
            curr_auc = auc(fpr, tpr)
            aucs.append(curr_auc)
            ax.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold {} (AUC={})'.format(i, round(curr_auc, 2)))

        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=0.8)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        mlflow.log_metric('roc auc', mean_auc)
        ax.plot(mean_fpr, mean_tpr, color='b',
                label='Mean ROC (AUC = {} +- {}'.format(round(mean_auc, 2), np.round(std_auc, 2)), lw=2, alpha=0.8)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2, label='std. dev.')
        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title='Receiver operating characteristic example')
        plt.xlabel(xlabel='False Positive Rate')
        plt.ylabel(ylabel='True Positive Rate')
        ax.legend(loc='lower right')
        fig_path = self.MID_RESULTS_DIR / 'roc_curve.png'
        fig.savefig(str(fig_path))
        mlflow.log_artifact(str(fig_path))

    def __log_precision_recall(self, score_folds):
        """

        :param score_folds: (n_samples, 2) array that holds the probabilities to each sample
        :return: No return value, the method logs the precision recall curve and logs the following metrics:
                precision, recall, average precision, f1 score
        """
        fig, ax = plt.subplots()
        y_real = []
        y_proba = []
        for i in range(self.cv.n_splits):
            curr_ground_truth_indices, curr_probs = score_folds[i]
            curr_ground_truth = self.y.iloc[curr_ground_truth_indices]
            curr_score = curr_probs[:, 1]
            precision, recall, _ = precision_recall_curve(curr_ground_truth, curr_score)
            curr_ap = average_precision_score(curr_ground_truth, curr_score)
            ax.plot(recall, precision, lw=1, alpha=0.3, label='Fold {} (AP={})'.format(i, round(curr_ap, 2)))
            y_real.append(curr_ground_truth)
            y_proba.append(curr_score)

        y_real = np.concatenate(y_real)
        y_proba = np.concatenate(y_proba)
        preds = np.round(y_proba).astype(int)
        precision, recall, _ = precision_recall_curve(y_real, y_proba)
        overall_ap = average_precision_score(y_real, y_proba)
        ax.plot(recall, precision, color='b',
                label='Overall AP  (AP = {})'.format(round(overall_ap, 2)), lw=2, alpha=0.8)
        mlflow.log_metric('average precision', overall_ap)
        recall = recall_score(y_real, preds)
        precision = precision_score(y_real, preds)
        f1 = f1_score(y_real, preds)
        mlflow.log_metric('recall', recall)
        mlflow.log_metric('precision', precision)
        mlflow.log_metric('f1 score', f1)
        plt.xlabel(xlabel='Recall')
        plt.ylabel(ylabel='Precision')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        ax.legend(loc='lower right')
        fig_path = self.MID_RESULTS_DIR / 'precision_recall_curve.png'
        fig.savefig(str(fig_path))
        mlflow.log_artifact(str(fig_path))

    def __log_errors(self, score_folds):
        X_copy = self.X.copy()
        num_patients = X_copy.shape[0]
        X_copy['Ground_Truth'] = list(self.y)
        correct_list = np.zeros(num_patients, dtype=bool)
        preds_list = np.zeros(num_patients, dtype=bool)
        score_list = np.zeros(num_patients)
        for i in range(self.cv.n_splits):
            curr_ground_truth_indices, curr_probs = score_folds[i]
            curr_ground_truth = self.y.iloc[curr_ground_truth_indices]
            curr_score = curr_probs[:, 1]
            curr_preds_int = np.round(curr_score).astype(int)
            curr_preds_bool = np.where(curr_preds_int == 1, True, False)
            preds_list[curr_ground_truth_indices] = curr_preds_bool
            correct_samples = (curr_ground_truth == curr_preds_bool)
            correct_list[curr_ground_truth_indices] = correct_samples
            score_list[curr_ground_truth_indices] = curr_score
        X_copy['Scores'] = score_list
        X_copy['Prediction'] = preds_list
        X_errors = X_copy[X_copy['Prediction'] != X_copy['Ground_Truth']]
        errors_path = self.MID_RESULTS_DIR / 'errors.csv'
        data_with_prediction_path = self.MID_RESULTS_DIR / 'data_with_prediction.csv'
        X_copy.to_csv(str(data_with_prediction_path))
        X_errors.to_csv(str(errors_path))
        mlflow.log_artifact(str(errors_path))
        mlflow.log_artifact(str(data_with_prediction_path))




    def __log_feature_importance(self):
        pass
    def run(self):
        """

        :return: No returns value, the function trains the model based on the pre-processing phases and calculated all
                the metrics of the run
        """
        if not self.MID_RESULTS_DIR.exists():
            self.MID_RESULTS_DIR.mkdir()
        if len(self.pre_process) == 0:
            full_model = make_pipeline(self.model)
        else:
            pipe = Pipeline(self.pre_process)
            full_model = make_pipeline(pipe, self.model)
        mlflow.sklearn.log_model(full_model, self.run_name)
        score_folds = []
        feature_names = list(self.X.columns)
        for i, (train, test) in enumerate(self.cv.split(self.X, self.y)):
            full_model.fit(self.X.iloc[train], self.y.iloc[train])
            y_probs = full_model.predict_proba(self.X.iloc[test])
            score_folds.append((test, y_probs))
        self.__log_roc(score_folds=score_folds)
        self.__log_precision_recall(score_folds=score_folds)
        self.__log_errors(score_folds=score_folds)
        print("End Run")
        mlflow.end_run()
        rmdir(dir_to_delete=self.MID_RESULTS_DIR)







