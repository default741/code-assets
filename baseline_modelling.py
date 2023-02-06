import numpy as np
import pandas as pd

from sklearn.metrics import make_scorer, fbeta_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, StratifiedKFold

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from concurrent.futures import ThreadPoolExecutor

from imblearn.pipeline import Pipeline as imbpipeline
from imblearn.over_sampling import SMOTE, ADASYN, KMeansSMOTE, SVMSMOTE, BorderlineSMOTE
from imblearn.under_sampling import TomekLinks, RepeatedEditedNearestNeighbours, NearMiss
from imblearn.combine import SMOTETomek, SMOTEENN

import joblib
import itertools


class _Utils:
    """Utility Class for Helping Fulctions

    Methods:
        _fbeta_eq_two_scorer: Calculates FBeta with beta=2 for weightage on Recall.
        _fbeta_eq_half_scorer: Calculates FBeta with beta=0.5 for weightage on Precision.
        _roc_threshold: Calculate Cutoff Threshold based on ROC Curve.
        _pr_threshold: Calculate Cutoff Threshold based on Precision-Recall Curve.
        _get_scorers: Returns Scoring Metrics.
    """

    @staticmethod
    def _fbeta_eq_two_scorer(yture: np.array, ypred: np.array) -> float:
        """
        Calculates FBeta with beta=2 for weightage on Recall.

        Args:
            yture (np.array): True Target Values
            ypred (np.array): Predicted Target Values

        Returns:
            float: FBeta Score
        """
        return fbeta_score(yture, ypred, beta=2)

    @staticmethod
    def _fbeta_eq_half_scorer(yture: np.array, ypred: np.array) -> float:
        """
        Calculates FBeta with beta=0.5 for weightage on Precision.

        Args:
            yture (np.array): True Target Values
            ypred (np.array): Predicted Target Values

        Returns:
            float: FBeta Score
        """
        return fbeta_score(yture, ypred, beta=0.5)

    @staticmethod
    def _get_scorers() -> dict:
        """
        Returns Scoring Metrics.

        Returns:
            dict: Scoring Metrics
        """

        return {
            'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall', 'f1': 'f1',
            'f2': make_scorer(_Utils._fbeta_eq_two_scorer, greater_is_better=True),
            'f0.5': make_scorer(_Utils._fbeta_eq_half_scorer, greater_is_better=True),
            'roc_auc': 'roc_auc',
        }


class BaselineModelling:

    def __init__(self) -> None:
        """Initialization Class.
        """

        self._sampling_methods = {
            'smote (up)': SMOTE(random_state=0),
            'adasyn (up)': ADASYN(random_state=0),
            'svm_smote (up)': SVMSMOTE(random_state=0),
            'boderline_smote (up)': BorderlineSMOTE(random_state=0),
            'k_means_smote (up)': KMeansSMOTE(random_state=0),

            'near_miss (down)': NearMiss(sampling_strategy='majority'),
            'renn (down)': RepeatedEditedNearestNeighbours(sampling_strategy='majority'),
            'tomek_links (down)': TomekLinks(sampling_strategy='majority'),

            'smote_tomek (combine)': SMOTETomek(random_state=0),
            'smote_enn (combine)': SMOTEENN(random_state=0)
        }

        self._models = {
            'random_forest': RandomForestClassifier,
            'gradient_boosting': GradientBoostingClassifier,
            'xgboost': XGBClassifier,
            'lightgbm': LGBMClassifier,
            'logistic_regression': LogisticRegression,
            'svm_rbf': SVC,
            'mlp_classifier': MLPClassifier
        }

    def load_data_pickle(self, file_path: str) -> dict:
        """Load Feature Selected Datasets.

        Args:
            file_path (str): Path of saved Data

        Returns:
            dict: Feature Selected Data
        """
        return joblib.load(file_path)

    def check_imbalance(self, target: np.ndarray, imbalance_class: int, imbalance_threshold: float = 0.1) -> bool:
        target_df = pd.DataFrame({'y_train': target})
        imbalance_percent = target_df[target_df['y_train']
                                      == imbalance_class].shape[0] / target_df.shape[0]

        if imbalance_percent < imbalance_threshold:
            return False

        return True

    def get_pipeline(self, sample_name: str, model_name: str, balanced: bool) -> object:
        if balanced:
            return imbpipeline(steps=[('model', self._models[model_name](random_state=0))])

        return imbpipeline(steps=[('sample', self._sampling_methods[sample_name]), ('model', self._models[model_name](random_state=0))])

    def get_pnc_list(self, feature_sets: list, balanced: str) -> list:
        if balanced:
            return list(itertools.product(feature_sets, [None], list(self._models.keys())))

        return list(itertools.product(feature_sets, list(self._sampling_methods.keys()), list(self._models.keys())))

    def run_cross_validation(self, pipeline, X_train, y_train, splits, run_name):
        return (cross_validate(pipeline, X_train, y_train, cv=splits, scoring=_Utils._get_scorers(), n_jobs=-1), run_name)

    def get_results(self, cv_results: dict, run_name: str, balanced: bool) -> list:
        feature_name, sample_name, model_name = run_name.split('?')

        if balanced:
            return [
                feature_name, 'balanced_data (None)', model_name,
                round(np.mean(cv_results['test_accuracy']) * 100, 2),
                round(np.mean(cv_results['test_precision']) * 100, 2),
                round(np.mean(cv_results['test_recall']) * 100, 2),
                round(np.mean(cv_results['test_f1']) * 100, 2),
                round(np.mean(cv_results['test_f2']) * 100, 2),
                round(np.mean(cv_results['test_f0.5']) * 100, 2),
                round(np.mean(cv_results['test_roc_auc']) * 100, 2)
            ]

        return [
            feature_name, sample_name, model_name,
            round(np.mean(cv_results['test_accuracy']) * 100, 2),
            round(np.mean(cv_results['test_precision']) * 100, 2),
            round(np.mean(cv_results['test_recall']) * 100, 2),
            round(np.mean(cv_results['test_f1']) * 100, 2),
            round(np.mean(cv_results['test_f2']) * 100, 2),
            round(np.mean(cv_results['test_f0.5']) * 100, 2),
            round(np.mean(cv_results['test_roc_auc']) * 100, 2)
        ]

    @staticmethod
    def complete_callback(future_obj: object) -> None:
        """Completion Callback Function
        Args:
            future_obj (object): Future Object with Function Pools
            func_name (str): Function Name
        """

        if future_obj.done():
            print(f"\tMethod Name: {future_obj.result()[1]} - COMPLETED.")

    def run_baseline_parallel(self, X_train: pd.DataFrame, y_train: np.ndarray, pnc_list: list, feature_list: dict, balanced: bool) -> list:
        baseline_list = list()
        results = list()

        for element in pnc_list:
            if feature_list[element[0]] == []:
                continue

            pipeline = self.get_pipeline(
                sample_name=element[1], model_name=element[2], balanced=balanced)

            splits = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
            run_name = f'{element[0]}?{element[1]}?{element[2]}'

            print(f'Method Name: {run_name} - ADDED')

            baseline_list.append(
                (self.run_cross_validation, (pipeline, X_train[feature_list[element[0]]], y_train, splits, run_name)))

        func_futures: list = list()

        print('\nSTATUS: PARALLEL FEATURE SELECTION')

        with ThreadPoolExecutor() as executor:
            for base_func, params in baseline_list:
                future_obj = executor.submit(base_func, *params)
                future_obj.add_done_callback(
                    BaselineModelling.complete_callback)

                func_futures.append(future_obj)

        for func_result in func_futures:
            cv_result, run_name = func_result.result()

            results.append(self.get_results(cv_results=cv_result,
                           run_name=run_name, balanced=balanced))

        return results

    def run_voting_classifier(
        self, X_train: pd.DataFrame, y_train: np.ndarray, feature_list: list, model_list: list, sample_list: list = None, balanced: bool = True
    ) -> None:

        results = list()

        print('\nRUNNING VOTING CLASSIFIER:')

        for select_feature in feature_list:
            print(f'\t\t{select_feature}')

            estimators_list: list = list()

            if balanced:
                estimators_list = [(f'{model_name}', self.get_pipeline(sample_name=None, model_name=model_name, balanced=balanced))
                                   for model_name in model_list]

            else:
                estimators_list = [(f'{model_name}-{sample_name}', self.get_pipeline(sample_name=sample_name, model_name=model_name, balanced=balanced))
                                   for model_name in model_list for sample_name in sample_list]

                print(estimators_list)

            voting_classifier = VotingClassifier(
                estimators=estimators_list, voting='soft')

            splits = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
            cv_result = cross_validate(
                voting_classifier, X_train[feature_list[select_feature]], y_train, cv=splits, scoring=_Utils._get_scorers(), n_jobs=-1)

            results.append(self.get_results(cv_results=cv_result,
                                            run_name=f'{select_feature}?{",".join(sample_list)}?{",".join(model_list)}', balanced=balanced))

        return results

    def save_results(self, results: list, sort_by: list, save_path: dict) -> None:
        """Save the Baseline Excel and the Best Results as a Joblib file.

        Args:
            results (list): Results Data List
            sort_by (list): Sort Baseline Report by KPI
            save_path (dict): Path to save results.
        """

        report = pd.DataFrame(data=results, columns=[
            'feature_set', 'sample_name', 'model', 'accuracy', 'precision', 'recall', 'f1', 'f2', 'f0.5', 'roc_auc'
        ]).sort_values(by=sort_by, ascending=False).reset_index(drop=True)

        best_result = report.iloc[:10][[
            'feature_set', 'sample_name', 'model']].to_dict(orient='list')
        best_result = {key: list(set(value))
                       for key, value in best_result.items()}

        report.to_excel(save_path['report'], index=False)
        joblib.dump({'best_result': best_result}, save_path['best_results'])

    def compile_baseline(
        self, file_path: str, balanced_data: bool, check_imbalance: bool, imbalance_class: int, imbalance_threshold: float, kpi_sorting: list,
        save_path: dict, enable_voting: bool, voting_model_list: list, voting_sample_list: list = []
    ) -> None:
        """Compiles all the Methods to run the Baseline Modelling.

        Args:
            file_path (str): Data File Path
            data_type (str): Balance of Data
            kpi_sorting (list): Sorting Metrics
            save_path (dict): Path to Save File
        """

        imbalance_class = int(imbalance_class)
        feature_list = dict()
        balanced = balanced_data

        feat_select_meta_data = self.load_data_pickle(
            file_path=file_path)['meta_data']

        X_train, X_test, y_train, y_test, selected_features = (
            feat_select_meta_data['X_train'], feat_select_meta_data['X_test'], feat_select_meta_data['y_train'],
            feat_select_meta_data['y_test'], feat_select_meta_data['selected_features']
        )

        feature_list = {f'{outer_key}-{inner_key}': feat_list for outer_key,
                        outer_value in selected_features.items() for inner_key, feat_list in outer_value.items()}

        if check_imbalance:
            balanced = self.check_imbalance(
                target=y_train, imbalance_class=imbalance_class, imbalance_threshold=imbalance_threshold)

        pnc_list = self.get_pnc_list(feature_sets=list(
            feature_list.keys()), balanced=balanced)

        results = self.run_baseline_parallel(
            X_train=X_train, y_train=y_train, pnc_list=pnc_list, feature_list=feature_list, balanced=balanced)

        if enable_voting:
            voting_results = self.run_voting_classifier(X_train=X_train, y_train=y_train, feature_list=feature_list,
                                                        model_list=voting_model_list, sample_list=voting_sample_list, balanced=balanced)

            results += voting_results

        self.save_results(
            results=results, sort_by=kpi_sorting, save_path=save_path)

        print(f'\nBaseline Modelling Finished\n{"-" * 100}')


if __name__ == '__main__':

    # Example Configuration File for Baseline Modelling.
    config = {
        'file_path': './data/feature_selected_data_v1.joblib',

        'balanced_data': False,
        'check_imbalance': False,
        'imbalance_class': 1,
        'imbalance_threshold': 0.1,

        'enable_voting': True,
        'voting_model_list': ['random_forest', 'lightgbm'],
        'voting_sample_list': ['smote (up)'],

        'kpi_sorting': ['roc_auc'],
        'save_path': {
            'report': './data/baseline_results_v1.xlsx',
            'best_results': './data/baseline_best_results_v1.joblib'
        }
    }

    BaselineModelling().compile_baseline(**config)
