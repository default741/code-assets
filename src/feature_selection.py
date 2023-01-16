from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from statsmodels.tools.tools import add_constant

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import (
    SelectFromModel, RFE, mutual_info_classif, f_classif, chi2, VarianceThreshold)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, ElasticNet

from boruta import BorutaPy
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import joblib
import warnings
import math

import pandas as pd
import numpy as np
import statsmodels.api as sm

warnings.filterwarnings('ignore')


class InvalidFileType(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class TargetFeatureException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class FeatureDropException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class InvalidSelectionMethod(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class DataException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class EmptyModelList(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class _Read_Data_File:
    """Protected class to Read Data with respect to its File Type. Currently Supports
    three file types namely CSV, EXCEL (xlsx), Parquet.

    Methods:
        _read_csv_type: Reads CSV file type
        _read_excel_type: Reads XLSX file type
        _read_parquet_type: Reads PARQUET file type
    """

    @staticmethod
    def _read_csv_type(file_path: str, params: dict) -> pd.DataFrame:
        """Reads CSV file type using Pandas Library (read_csv).

        Args:
            file_path (str): Path to read the file from.
            params (dict): Extra Parameters for the Method

        Returns:
            pd.DataFrame: Raw Data File
        """
        return pd.read_csv(file_path, **params)

    @staticmethod
    def _read_excel_type(file_path: str, params: dict) -> pd.DataFrame:
        """Reads XLSX file type using Pandas Library (read_excel).

        Args:
            file_path (str): Path to read the file from.
            params (dict): Extra Parameters for the Method

        Returns:
            pd.DataFrame: Raw Data File
        """
        return pd.read_excel(file_path, **params)

    @staticmethod
    def _read_parquet_type(file_path: str, params: dict) -> pd.DataFrame:
        """Reads Parquet file type using Pandas Library (read_parquet).

        Args:
            file_path (str): Path to read the file from.
            params (dict): Extra Parameters for the Method

        Returns:
            pd.DataFrame: Raw Data File
        """
        return pd.read_parquet(file_path, **params)


class _Utils:

    @staticmethod
    def _calculate_vif(data: pd.DataFrame, add_const: bool = True) -> pd.DataFrame:
        """Calculates Variance Inflation Factor for Feature Multicollinearity

        Args:
            data (pd.DataFrame): Input Data

        Raises:
            DataException: If Dataframe is empty.

        Returns:
            pd.DataFrame: Features with their VIF Values.
        """

        if data.empty:
            raise DataException('Dataframe cannot be empty.')

        if add_const:
            data = add_constant(data)

        return pd.DataFrame({'Feature': data.columns.values, 'VIF': [vif(data.values, idx) for idx in range(data.shape[1])]})

    @staticmethod
    def _calculate_pvalues(data: pd.DataFrame, target: np.array, fit_method: str = None) -> pd.DataFrame:
        """Calculates and Returns P-Values from Data.

        Args:
            data (pd.DataFrame): Input Data
            target (np.array): Target Data

        Raises:
            DataException: If Dataframe is empty.
            DataException: Input Data and Target Data should have same Length.

        Returns:
            pd.DataFrame: Dataframe containing P-Values.
        """

        model = None

        if data.empty:
            raise DataException('Dataframe cannot be empty.')

        if data.shape[0] != target.shape[0]:
            raise DataException(
                'Input Data and Target Data should have same Length.')

        if fit_method is not None:
            model = sm.Logit(target, data).fit(disp=0, method=fit_method)
        else:
            model = sm.Logit(target, data).fit(disp=0)

        return pd.DataFrame({'feature': model.pvalues.index, 'p_value': model.pvalues.values})

    @staticmethod
    def _encode_target_variables(data: pd.DataFrame, target_feature: str) -> tuple:
        """Encodes the target feature if not already encoded

        Args:
            data (pd.DataFrame): Input data with the target feature
            target_feature (str): Target Feature Name

        Returns:
            tuple: Dataframe with target feature encoded, encoder object
        """

        encoder = LabelEncoder().fit(data[target_feature])
        data[target_feature] = encoder.transform(data[target_feature])

        return (data, encoder)


class _Select_Methods:

    @staticmethod
    def _anova_f_value_selection(data: pd.DataFrame, target: pd.DataFrame, kwargs: dict) -> tuple:
        """Feature Selection Method using ANOVA-F Value

        Args:
            data (pd.DataFrame): Input Data
            target (pd.DataFrame): Target Data
            kwargs (dict): Extra Key-Word Arguments

        Returns:
            tuple: Selected Features Dictionary with the Function Name
        """

        num_feat = kwargs['num_feat'] if kwargs['num_feat'] is not None else 15
        anova_f_values, anova_f_pvalues = f_classif(data, target)

        feature_list = pd.DataFrame(
            {'features': data.columns, 'anova_f_values': anova_f_values, 'anova_f_pvalues': anova_f_pvalues})

        feature_list = feature_list[feature_list['anova_f_pvalues'] > 0.05].sort_values(by=[
                                                                                        'anova_f_values'])

        return ({'anova_f_value': list(data[feature_list['features'].iloc[:num_feat]].columns)}, 'anova_f_value_selection')

    @staticmethod
    def _mutual_info_classif_selection(data: pd.DataFrame, target: pd.DataFrame, kwargs: dict) -> tuple:
        """Feature Selection Method using Mutual Information Classifier

        Args:
            data (pd.DataFrame): Input Data
            target (pd.DataFrame): Target Data
            kwargs (dict): Extra Key-Word Arguments

        Returns:
            tuple: Selected Features Dictionary with the Function Name
        """

        num_feat = kwargs['num_feat'] if kwargs['num_feat'] is not None else 15
        mutual_info = mutual_info_classif(data, target)

        feature_list = pd.DataFrame(
            {'features': data.columns, 'mutual_info': mutual_info})

        feature_list = feature_list[feature_list['mutual_info'] > 0].sort_values(by=[
                                                                                 'mutual_info'])

        return ({'mutual_info_classif': list(data[feature_list['features'].iloc[:num_feat]].columns)}, 'mutual_info_classif_selection')

    @staticmethod
    def _logit_selection(data: pd.DataFrame, target: pd.DataFrame, kwargs: dict) -> tuple:
        """Selection Feature Using Logit Model.

        Args:
            data (pd.DataFrame): Input Data
            target (np.array): Target Data
            kwargs (dict): Extra Key-Word Arguments

        Raises:
            FeatureDropException: All Features Dropped.

        Returns:
            tuple: Selected Features Dictionary with the Function Name
        """

        p_values = None

        if 'fit_method' in kwargs and kwargs['fit_method'] is not None:
            p_values = _Utils._calculate_pvalues(
                data=data, target=target, fit_method=kwargs['fit_method'])
        else:
            p_values = _Utils._calculate_pvalues(data=data, target=target)

        while p_values['p_value'].max() > 0.05:
            id_max = p_values[['p_value']].idxmax()
            feature_to_drop = str(p_values.iloc[id_max]['feature'].values[0])

            data = data.drop(columns=[feature_to_drop])

            if data.shape[1] == 0:
                raise FeatureDropException(
                    'All Features Dropped. Suggestion - Don\'t use Logit Selection or try a different fit method.')

            if 'fit_method' in kwargs and kwargs['fit_method'] is not None:
                p_values = _Utils._calculate_pvalues(
                    data=data, target=target, fit_method=kwargs['fit_method'])
            else:
                p_values = _Utils._calculate_pvalues(data=data, target=target)

        return ({'logit': list(data.columns)}, 'logit_selection')

    @staticmethod
    def _permutation_impt_selection(data: pd.DataFrame, target: pd.DataFrame, kwargs: dict) -> tuple:
        """Feature Selection Using Permutation Importance with Random Forest Model.

        Args:
            data (pd.DataFrame): Input Data
            target (np.array): Target Data
            kwargs (dict): Extra Key-Word Arguments

        Raises:
            EmptyModelList: No Model List Parameter in Key Word Arguments.
            EmptyModelList: Specified Model Type not in List.

        Returns:
            tuple: Selected Features Dictionary with the Function Name
        """

        perm_impt_dict = dict()

        model_list = {
            'random_forest': RandomForestClassifier(random_state=0, n_jobs=-1),
            'catboost': CatBoostClassifier(random_state=0, n_estimators=100, verbose=False),
            'xgboost': XGBClassifier(random_state=0, n_jobs=-1),
            'lightgbm': LGBMClassifier(random_state=0, n_jobs=-1),
            'gradient_boosting': GradientBoostingClassifier(random_state=0),
            'logistic_regression': LogisticRegression(random_state=0, n_jobs=-1)
        }

        num_feat = kwargs['num_feat'] if kwargs['num_feat'] is not None else 15

        if 'model_list' in kwargs:
            models = (kwargs['model_list'] if kwargs['model_list'][0].lower(
            ) != 'all' and kwargs['model_list'] != [] else model_list.keys())

            for model_type in models:
                model = None

                if model_type in model_list:
                    model = model_list.get(model_type).fit(data, target)
                else:
                    raise EmptyModelList(
                        f'No Model Type {model_type} in Model List.')

                perm_impt = permutation_importance(
                    model, data, target, n_repeats=10, random_state=0, n_jobs=-1)

                perm_df = pd.DataFrame({'feature': data.columns.values, 'avg_impt': perm_impt.importances_mean}).sort_values(
                    by=['avg_impt'], ascending=False).reset_index(drop=True)

                data = data[perm_df[perm_df['avg_impt'] > 0]
                            ['feature'].iloc[:num_feat].values]

                perm_impt_dict[model_type] = list(data.columns)

        else:
            raise EmptyModelList(
                'No Model List Parameter in Key Word Arguments.')

        return (perm_impt_dict, 'permutation_impt_selection')

    @staticmethod
    def _recursive_feature_elimination(data: pd.DataFrame, target: pd.DataFrame, kwargs: dict) -> tuple:
        """Selection Feature Using Recursive Feature Elimination with Random Forest Model. Number of features defaults to 25.

        Args:
            data (pd.DataFrame): Input Data
            target (np.array): Target Data
            kwargs (dict): Extra Key-Word Arguments

        Raises:
            EmptyModelList: No Model List Parameter in Key Word Arguments.
            EmptyModelList: Specified Model Type not in List.

        Returns:
            tuple: Selected Features Dictionary with the Function Name
        """

        rfe_impt_dict = dict()

        model_list = {
            'random_forest': RandomForestClassifier(random_state=0, n_jobs=-1),
            'catboost': CatBoostClassifier(random_state=0, n_estimators=100, verbose=False),
            'xgboost': XGBClassifier(random_state=0, n_jobs=-1),
            'lightgbm': LGBMClassifier(random_state=0, n_jobs=-1),
            'gradient_boosting': GradientBoostingClassifier(random_state=0),
            'logistic_regression': LogisticRegression(random_state=0, n_jobs=-1)
        }

        num_feat = kwargs['num_feat'] if kwargs['num_feat'] is not None else 15
        step_value = math.ceil(len(
            data.columns) / num_feat) if kwargs['step_value'] is None else kwargs['step_value']

        if 'model_list' in kwargs:
            models = (kwargs['model_list'] if kwargs['model_list'][0].lower(
            ) != 'all' and kwargs['model_list'] != [] else model_list.keys())

            for model_type in models:
                model = None

                if model_type in model_list:
                    model = model_list.get(model_type).fit(data, target)
                else:
                    raise EmptyModelList(
                        f'No Model Type {model_type} in Model List.')

                rfe_model = RFE(estimator=model, step=step_value,
                                n_features_to_select=num_feat, verbose=False).fit(data, target)

                data = data[data.columns[rfe_model.support_]]
                rfe_impt_dict[model_type] = list(data.columns)

        else:
            raise EmptyModelList(
                'No Model List Parameter in Key Word Arguments.')

        return (rfe_impt_dict, 'recursive_feature_elimination')

    @staticmethod
    def _model_based_importance(data: pd.DataFrame, target: pd.DataFrame, kwargs: dict) -> tuple:
        """Selection Feature Using Random Forest Model's Feature Importance.

        Args:
            data (pd.DataFrame): Input Data
            target (np.array): Target Data
            kwargs (dict): Extra Key-Word Arguments

        Raises:
            EmptyModelList: No Model List Parameter in Key Word Arguments.
            EmptyModelList: Specified Model Type not in List.

        Returns:
            tuple: Selected Features Dictionary with the Function Name
        """

        mbi_feat = dict()

        model_list = {
            'random_forest': RandomForestClassifier(random_state=0, n_jobs=-1),
            'xgboost': XGBClassifier(random_state=0, n_jobs=-1),
            'catboost': CatBoostClassifier(random_state=0, verbose=False, n_estimators=100),
            'lightgbm': LGBMClassifier(random_state=0, n_jobs=-1),
        }

        num_feat = kwargs['num_feat'] if kwargs['num_feat'] is not None else 15

        if 'model_list' in kwargs:
            models = (kwargs['model_list'] if kwargs['model_list'][0].lower(
            ) != 'all' and kwargs['model_list'] != [] else model_list.keys())

            for model_type in models:
                model = None

                if model_type in model_list:
                    model = model_list.get(model_type).fit(data, target)
                else:
                    raise EmptyModelList(
                        f'No Model Type {model_type} in Model List.')

                impt_df = pd.DataFrame({'feature': data.columns, 'avg_impt': model.feature_importances_}).sort_values(
                    by=['avg_impt'], ascending=False).reset_index(drop=True)

                data = data[impt_df.iloc[:num_feat]['feature'].values]

                mbi_feat[model_type] = list(data.columns)

        else:
            raise EmptyModelList(
                'No Model List Parameter in Key Word Arguments.')

        return (mbi_feat, 'model_based_importance')

    @staticmethod
    def _regularization_selection(data: pd.DataFrame, target: pd.DataFrame, kwargs: dict) -> tuple:
        """Selection Feature Using Lasso Feature Selection.

        Args:
            data (pd.DataFrame): Input Data
            target (np.array): Target Data
            kwargs (dict): Extra Key-Word Arguments

        Raises:
            EmptyModelList: No Model List Parameter in Key Word Arguments.
            EmptyModelList: Specified Model Type not in List.

        Returns:
            tuple: Selected Features Dictionary with the Function Name
        """

        reg_feat = dict()

        model_list = {
            'lasso': LogisticRegression(penalty='l1', random_state=0, n_jobs=-1, C=0.1, solver='saga'),
            'ridge': RidgeClassifier(alpha=1.0, random_state=0),
            'elasticnet': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=0)
        }

        num_feat = kwargs['num_feat'] if kwargs['num_feat'] is not None else 15

        if 'model_list' in kwargs:
            models = (kwargs['model_list'] if kwargs['model_list'][0].lower(
            ) != 'all' and kwargs['model_list'] != [] else model_list.keys())

            for model_type in models:
                model = None

                if model_type in model_list:
                    model = model_list.get(model_type).fit(data, target)
                else:
                    raise EmptyModelList(
                        f'No Model Type {model_type} in Model List.')

                selector = SelectFromModel(
                    estimator=model, max_features=num_feat, threshold=-np.inf).fit(data, target)

                data = pd.DataFrame(data=selector.transform(
                    data), columns=data.columns[selector.get_support()])
                reg_feat[model_type] = list(data.columns)

        else:
            raise EmptyModelList(
                'No Model List Parameter in Key Word Arguments.')

        return (reg_feat, 'regularization_selection')

    @staticmethod
    def _boruta_selection(data: pd.DataFrame, target: pd.DataFrame, kwargs: dict) -> tuple:
        """Selection Feature Using Boruta with Random Forest Model.

        Args:
            data (pd.DataFrame): Input Data
            target (np.array): Target Data
            kwargs (dict): Extra Key-Word Arguments

        Raises:
            EmptyModelList: No Model List Parameter in Key Word Arguments.
            EmptyModelList: Specified Model Type not in List.

        Returns:
            tuple: Selected Features Dictionary with the Function Name
        """

        boruta_impt_dict = dict()

        model_list = {
            'random_forest': RandomForestClassifier(random_state=0, n_jobs=-1),
            'xgboost': XGBClassifier(random_state=0, n_jobs=-1),
            'lightgbm': LGBMClassifier(random_state=0, n_jobs=-1),
            'gradient_boosting': GradientBoostingClassifier(random_state=0),
        }

        if 'model_list' in kwargs:
            models = (kwargs['model_list'] if kwargs['model_list'][0].lower(
            ) != 'all' and kwargs['model_list'] != [] else model_list.keys())

            for model_type in models:
                model = None

                if model_type in model_list:
                    model = model_list.get(model_type)
                else:
                    raise EmptyModelList(
                        f'No Model Type {model_type} in Model List.')

                boruta_model = BorutaPy(
                    estimator=model, random_state=0, verbose=False).fit(data.values, target)

                selected_data = data[data.columns[boruta_model.support_].to_list(
                ) + data.columns[boruta_model.support_weak_].to_list()]

                boruta_impt_dict[model_type] = list(selected_data.columns)

        else:
            raise EmptyModelList(
                'No Model List Parameter in Key Word Arguments.')

        return (boruta_impt_dict, 'boruta_selection')

    @staticmethod
    def _sequencial_forward_selection(data: pd.DataFrame, target: pd.DataFrame, kwargs: dict) -> tuple:
        """Selection Feature Using Sequencial Forward Selection with Random Forest Model.

        Args:
            data (pd.DataFrame): Input Data
            target (np.array): Target Data
            kwargs (dict): Extra Key-Word Arguments

        Raises:
            EmptyModelList: No Model List Parameter in Key Word Arguments.
            EmptyModelList: Specified Model Type not in List.

        Returns:
            tuple: Selected Features Dictionary with the Function Name
        """

        sfs_impt_dict = dict()

        model_list = {
            'random_forest': RandomForestClassifier(random_state=0, n_jobs=-1),
            'xgboost': XGBClassifier(random_state=0, n_jobs=-1),
            'lightgbm': LGBMClassifier(random_state=0, n_jobs=-1),
            'gradient_boosting': GradientBoostingClassifier(random_state=0),
        }

        num_feat = kwargs['num_feat'] if kwargs['num_feat'] is not None else 15
        scoring_metric = kwargs['scoring_metric'] if kwargs['scoring_metric'] is not None else 'roc_auc'

        if 'model_list' in kwargs:
            models = (kwargs['model_list'] if kwargs['model_list'][0].lower(
            ) != 'all' and kwargs['model_list'] != [] else model_list.keys())

            for model_type in models:
                model = None

                if model_type in model_list:
                    model = model_list.get(model_type).fit(data, target)
                else:
                    raise EmptyModelList(
                        f'No Model Type {model_type} in Model List.')

                fs_model = sfs(model, k_features=num_feat, verbose=False, forward=True,
                               scoring=scoring_metric, cv=5, n_jobs=-1).fit(data, target)

                metrics = fs_model.get_metric_dict()
                cur_max, itr = 0, 0

                for i in range(1, num_feat + 1):
                    try:
                        if metrics[i]['avg_score'] > cur_max:
                            cur_max, itr = metrics[i]['avg_score'], i

                    except Exception as e:
                        print(f'Exception Was Raised: {e}')

                feat_list = list(metrics[itr]['feature_names'])
                sfs_impt_dict[model_type] = list(data[list(feat_list)].columns)

        else:
            raise EmptyModelList(
                'No Model List Parameter in Key Word Arguments.')

        return (sfs_impt_dict, 'sequencial_forward_selection')


class FeatureSelection:
    """Transforms Raw Data to Model Injectable Data.

    Methods:
        __init__: Class Initialization Method.
        read_data: Reads Raw Data.
        drop_low_variance_features: Drop Freatures having very low variance in the data.
    """

    def __init__(self) -> None:
        """Class Initialization Method.
        """

        self._read_data = {
            'csv': _Read_Data_File._read_csv_type,
            'xlsx': _Read_Data_File._read_excel_type,
            'parquet': _Read_Data_File._read_parquet_type
        }

        self._selection_methods = {
            'anova_f_value_selection': _Select_Methods._anova_f_value_selection,
            'mutual_info_classif_selection': _Select_Methods._mutual_info_classif_selection,
            'logit_selection': _Select_Methods._logit_selection,
            'permutation_impt_selection': _Select_Methods._permutation_impt_selection,
            'recursive_feature_elimination': _Select_Methods._recursive_feature_elimination,
            'model_based_importance': _Select_Methods._model_based_importance,
            'regularization_selection': _Select_Methods._regularization_selection,
            'boruta_selection': _Select_Methods._boruta_selection,
            'sequencial_forward_selection': _Select_Methods._sequencial_forward_selection
        }

    def read_data(self, file_type: str, file_path: str, target_feature: str = None, encode_target: bool = True, **kwargs) -> tuple:
        """Reads Raw Data

        Args:
            file_path (str): File Path for Raw Data
            file_type (str): File Extension
            target_feature (str, optional): Target Label. Defaults to None.

        Raises:
            InvalidFileType: File Type Invalid. Not present in the specified List of types.
            TargetFeatureException: Target Feature Name not Provided. Please Provide a valid Target feature Name.
            TargetFeatureException: Target Feature not found in Raw Data Columns.

        Returns:
            tuple: Input Data and Target Data
        """

        raw_data = pd.DataFrame()
        encoder = None

        if file_type not in self._read_data.keys():
            raise InvalidFileType(
                'File Type Invalid. Not present in the specified List of types.')

        if target_feature is None:
            raise TargetFeatureException(
                'Target Feature Name not Provided. Please Provide a valid Target feature Name.')

        try:
            raw_data = self._read_data[file_type](
                file_path=file_path, params=kwargs)

        except Exception as E:
            print(
                f'Error occured While reading the File. Please Check the parameters carefully. {E}')

        if target_feature not in raw_data.columns:
            raise TargetFeatureException(
                'Target Feature not found in Raw Data Columns.')

        if encode_target:
            raw_data, encoder = _Utils._encode_target_variables(
                data=raw_data, target_feature=target_feature)

        input_data = raw_data.drop(columns=[target_feature])
        target_data = raw_data[target_feature].values

        file_name = file_path.split('/')[-1].split('.')[0]

        print(f'READING FILE NAME: {file_name}')
        print(
            f'SHAPE: ROWS - {raw_data.shape[0]}, COLUMNS - {raw_data.shape[1] - 1}')
        print(
            f'TARGET LABEL: {target_feature}, ENCODER CLASS: {encoder.classes_}')

        return (input_data, target_data, encoder)

    def drop_low_variance_features(self, data: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """Drop Freatures having very low variance in the data.

        Args:
            data (pd.DataFrame): Input Data
            threshold (float, optional): Variance Threshold. Defaults to 0.5.

        Returns:
            pd.DataFrame: Low Varinace Dropped Data
        """

        init_columns = set(data.columns)

        var_thresh_object = VarianceThreshold(threshold=threshold)
        select_feat = var_thresh_object.fit_transform(data)

        data = pd.DataFrame(
            data=select_feat, columns=data.columns[var_thresh_object.get_support()])

        print(
            f'\nAFTER DROPPING LOW VARIANCE FEATURES (THRESHOLD - {threshold}):')
        print(f'SHAPE: ROWS - {data.shape[0]}, COLUMNS - {data.shape[1]}')
        print(f'FEATURES DROPPED: {list(init_columns - set(data.columns))}')

        return data

    def drop_high_correlated_features(self, data: pd.DataFrame, threshold: float = 0.7, corr_method: str = 'pearson') -> pd.DataFrame:
        """Drop Freatures having very high correlation between each other.

        Args:
            data (pd.DataFrame): Input Data
            threshold (float, optional): Correlation Threshold. Defaults to 0.7.
            corr_method (str, optional): Correlation Method. Defaults to 'pearson'.

        Returns:
            pd.DataFrame: Dataframe with High Correlated Features Dropped.
        """

        init_columns = set(data.columns)
        corr_matrix = data.corr(method=corr_method).abs()

        corr_upper_matrix = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        cols_to_drop = [col for col in corr_upper_matrix.columns if any(
            corr_upper_matrix[col] > threshold)]

        data = data.drop(columns=cols_to_drop).reset_index(drop=True)

        print(
            f'\nAFTER DROPPING HIGH CORRELATED FEATURES (THRESHOLD - {threshold}):')
        print(f'SHAPE: ROWS - {data.shape[0]}, COLUMNS - {data.shape[1]}')
        print(f'FEATURES DROPPED: {list(init_columns - set(data.columns))}')

        return data

    def drop_multicolliner_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Drops Features having High Multicollinearity from the data.

        Args:
            data (pd.DataFrame): Input Data

        Raises:
            FeatureDropException: All Features Dropped.

        Returns:
            pd.DataFrame: Output Data with No Collinear Features
        """

        print('\nRUNNING: DROP MULTI-COLLINEAR FEATURES')

        init_columns = set(data.columns)
        vif_df = _Utils._calculate_vif(data=data)

        while vif_df['VIF'].max() > 5:
            try:
                id_max = vif_df[['VIF']].idxmax()
                feature_to_drop = str(vif_df.iloc[id_max]['Feature'].values[0])

                print(
                    f'\tFeature: {feature_to_drop}, VIF Value: {round(float(vif_df.iloc[id_max]["VIF"].values[0]), 2)}')

                data = data.drop(columns=[feature_to_drop])

                if data.shape[1] == 0:
                    raise FeatureDropException(
                        'All Features Dropped while dropping Multicollinear Features.')

                vif_df = _Utils._calculate_vif(data=data)

            except Exception as e:
                print(f'Error Occured While Calculating VIF - {e}')

            finally:
                break

        print(
            f'\nAFTER DROPPING MULTI-COLLINEAR FEATURES:')
        print(f'SHAPE: ROWS - {data.shape[0]}, COLUMNS - {data.shape[1]}')
        print(f'FEATURES DROPPED: {list(init_columns - set(data.columns))}')

        return data

    def select_features(self, data: pd.DataFrame, target: np.array, conf: list) -> dict:
        """Runs Feature Selection Pipeline to get the optimal features for Modelling.

        Args:
            data (pd.DataFrame): Input Data.
            target (np.array): Target Data
            conf (list): Configuration for Feature Selection.

        Raises:
            InvalidSelectionMethod: Invalid Feature Selection Method.

        Returns:
            dict: Output Dataset with Selected Features.
        """

        print('\nRUNNING: FEATURE SELECTION')

        feat_dict = dict()

        for select_method in conf:
            method_name = select_method['select_method']

            print(f'\tMethod Name: {method_name}')

            if method_name in self._selection_methods.keys():
                feat_dict[method_name] = self._selection_methods[method_name](
                    data.copy(), target, select_method['params'])

            else:
                raise InvalidSelectionMethod(
                    'Invalid Feature Selection Method.')

        return feat_dict

    @staticmethod
    def complete_callback(future_obj: object) -> None:
        """Completion Callback Function

        Args:
            future_obj (object): Future Object with Function Pools
            func_name (str): Function Name
        """

        if future_obj.done():
            print(f"\tMethod Name: {future_obj.result()[1]} - COMPLETED.")

    def parallel_select_features(self, data: pd.DataFrame, target: np.array, conf: list) -> dict:
        """Runs Feature Selection Pipeline to get the optimal features for Modelling.

        Args:
            data (pd.DataFrame): Input Data.
            target (np.array): Target Data
            conf (list): Configuration for Feature Selection.

        Raises:
            InvalidSelectionMethod: Invalid Feature Selection Method.

        Returns:
            dict: Output Dataset with Selected Features.
        """

        print('\nRUNNING: PARALLEL FEATURE SELECTION')

        feat_dict: dict = dict()
        select_function_list: list = list()

        for select_method in conf:
            method_name = select_method['select_method']

            print(f'\tMethod Name: {method_name}')

            if method_name in self._selection_methods.keys():
                select_function_list.append(
                    (self._selection_methods[method_name], (data.copy(), target, select_method['params'])))

            else:
                raise InvalidSelectionMethod(
                    'Invalid Feature Selection Method.')

        func_futures: list = list()

        print('\nSTATUS: PARALLEL FEATURE SELECTION')

        with ThreadPoolExecutor() as executor:
            for select_func, params in select_function_list:
                future_obj = executor.submit(select_func, *params)
                future_obj.add_done_callback(
                    FeatureSelection.complete_callback)

                func_futures.append(future_obj)

        for func_result in func_futures:
            selected_feat_dict, func_name = func_result.result()
            feat_dict[func_name] = selected_feat_dict

        return feat_dict

    def save_data(self, meta_data: object, path: str) -> None:
        """Save the Dataset Pipeline

        Args:
            data_dict (object): Feature Selected Data Dict
            path (str): Saving Path
        """

        print('\nSAVING JOBLIB FILE (WITH DATA AND FEATURES)...')

        joblib.dump({'process_type': path.split(
            '/')[-1].split('.')[0], 'meta_data': meta_data}, path)

    def compile_selection(
        self, file_path: str, file_type: str, target_feature: str, test_size: float, save_path: str, drop_multicolliner_features: bool = True,
        drop_low_variance_features: bool = True, variance_thresh: float = 0.5, feature_select_conf: list = [], drop_high_corr_features: bool = True,
        corr_threshold: float = 0.7, corr_method: str = 'pearson', run_parallel: bool = True
    ) -> dict:
        """Compile and Runs the Feature Selection Pipeline

        Args:
            file_path (str): Raw Data File Path
            file_type (str): File Extention
            target_feature (str): Target Label
            test_size (float): Test Size Percentage
            save_path (str): Path to save Pipeline
            drop_multicolliner_features (bool): To Drop Multicollinear Features or Not.
            drop_low_variance_features (bool): To Drop Low Variance Features or Not.

        Returns:
            dict: Final Data Dict
        """

        print(f'\nData Transformation Started.\n')

        data, target, encoder = self.read_data(
            file_path=file_path, file_type=file_type, target_feature=target_feature)

        X_train, X_test, y_train, y_test = train_test_split(
            data, target, test_size=test_size, shuffle=True, stratify=target, random_state=0)

        saving_dict = {
            'X_train': X_train.copy(), 'X_test': X_test.copy(), 'y_train': y_train, 'y_test': y_test, 'encoder': encoder
        }

        if drop_low_variance_features:
            X_train = self.drop_low_variance_features(
                data=X_train.copy(), threshold=variance_thresh)

        if drop_high_corr_features:
            X_train = self.drop_high_correlated_features(
                data=X_train.copy(), threshold=corr_threshold, corr_method=corr_method)

        if drop_multicolliner_features:
            X_train = self.drop_multicolliner_features(data=X_train.copy())

        selection_dict = dict()

        if run_parallel:
            selection_dict = self.parallel_select_features(
                data=X_train.copy(), target=y_train, conf=feature_select_conf)
        else:
            selection_dict = self.select_features(
                data=X_train.copy(), target=y_train, conf=feature_select_conf)

        saving_dict['selected_features'] = selection_dict

        self.save_data(meta_data=saving_dict, path=save_path)

        print(f'\nData Transformation Finished.\n{"-" * 100}')

        return selection_dict


if __name__ == '__main__':

    # Example Configuration File for Feature Selection.
    config = {
        'file_path': './data/transformed_data_v1.csv',
        'file_type': 'csv',
        'target_feature': 'target_label',
        'run_parallel': True,

        'drop_low_variance_features': True,
        'variance_thresh': 0.3,

        'drop_high_corr_features': True,
        'corr_threshold': 0.8,
        'corr_method': 'pearson',

        'drop_multicolliner_features': True,

        'feature_select_conf': [
            {
                'select_method': 'anova_f_value_selection',
                'params': {
                    'num_feat': 15
                }
            },

            {
                'select_method': 'mutual_info_classif_selection',
                'params': {
                    'num_feat': 15
                }
            },

            {
                'select_method': 'logit_selection',
                'params': {
                    'fit_method': None
                }
            },

            {
                'select_method': 'permutation_impt_selection',
                'params': {
                    'model_list': ['all'],
                    'num_feat': 15
                }
            },

            {
                'select_method': 'recursive_feature_elimination',
                'params': {
                    'model_list': ['all'],
                    'num_feat': 15,
                    'step_value': None
                }
            },

            {
                'select_method': 'model_based_importance',
                'params': {
                    'model_list': ['all'],
                    'num_feat': 15,
                }
            },

            {
                'select_method': 'regularization_selection',
                'params': {
                    'model_list': ['all'],
                    'num_feat': 15,
                }
            },

            {
                'select_method': 'boruta_selection',
                'params': {
                    'model_list': ['random_forest', 'lightgbm'],
                }
            },

            {
                'select_method': 'sequencial_forward_selection',
                'params': {
                    'model_list': ['random_forest', 'lightgbm'],
                    'num_feat': 15,
                    'scoring_metric': None
                }
            }
        ],
        'test_size': 0.2,
        'save_path': './data/feature_selected_data_v1.joblib'
    }

    feature_selection = FeatureSelection().compile_selection(**config)

    import json

    with open('feat_select.json', 'w') as file:
        file.write(json.dumps(feature_selection, indent=4))
