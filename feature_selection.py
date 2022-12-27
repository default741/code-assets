from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from statsmodels.tools.tools import add_constant

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.linear_model import LogisticRegression, RidgeClassifier, ElasticNet

from boruta import BorutaPy
from mlxtend.feature_selection import SequentialFeatureSelector as sfs

import joblib

import pandas as pd
import numpy as np
import statsmodels.api as sm


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
            ValueError: If Dataframe is empty.

        Returns:
            pd.DataFrame: Features with their VIF Values.
        """

        if data.empty:
            raise ValueError('Dataframe cannot be empty.')

        if add_const:
            data = add_constant(data)

        return pd.DataFrame({'Feature': data.columns.values, 'VIF': [vif(data.values, idx) for idx in range(data.shape[1])]})

    @staticmethod
    def _calculate_pvalues(data: pd.DataFrame, target: np.array, fit_method: str = 'Newton-Raphson') -> pd.DataFrame:
        """Calculates and Returns P-Values from Data.

        Args:
            data (pd.DataFrame): Input Data
            target (np.array): Target Data

        Raises:
            ValueError: If Dataframe is empty.
            ValueError: Input Data and Target Data should have same Length.

        Returns:
            pd.DataFrame: Dataframe containing P-Values.
        """

        if data.empty:
            raise ValueError('Dataframe cannot be empty.')

        if data.shape[0] != target.shape[0]:
            raise ValueError(
                'Input Data and Target Data should have same Length.')

        model = sm.Logit(target, data).fit(disp=0, method=fit_method)

        return pd.DataFrame({'feature': model.pvalues.index, 'p_value': model.pvalues.values})


class _Select_Methods:

    # TODO: Add Correlation based FS
    # TODO: Add Variance based FS
    # TODO: Add Chi-Squared Test based FS
    # TODO: Add ANOVA F-Value based FS
    # TODO: Add Mutual Info Classif based FS

    @staticmethod
    def _logit_selection(data: pd.DataFrame, target: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Selection Feature Using Logit Model.

        Args:
            data (pd.DataFrame): Input Data
            target (np.array): Target Data

        Raises:
            ValueError: All Features Dropped.

        Returns:
            pd.DataFrame: Feature Selected DataFrame.
        """

        print(f'\tRunning Logit Feature Selection')

        p_values = None

        if 'fit_method' in kwargs and kwargs['fit_method'] is not None:
            p_values = _Utils._calculate_pvalues(
                data=data, target=target, fit_method=kwargs['fit_method'])
        else:
            p_values = _Utils._calculate_pvalues(data=data, target=target)

        while p_values['p_value'].max() > 0.05:
            id_max = p_values[['p_value']].idxmax()
            feature_to_drop = str(p_values.iloc[id_max]['feature'].values[0])

            print(
                f'\t\tFeature Dropped: {feature_to_drop}; p-value: {p_values.iloc[id_max]["p_value"].values[0]}')

            data = data.drop(columns=[feature_to_drop])

            if data.shape[1] == 0:
                raise ValueError(
                    'All Features Dropped. Suggestion - Don\'t use Logit Selection or try a different fit method.')

            if 'fit_method' in kwargs:
                p_values = _Utils._calculate_pvalues(
                    data=data, target=target, fit_method=kwargs['fit_method'])
            else:
                p_values = _Utils._calculate_pvalues(data=data, target=target)

        data['target_label'] = target

        return data

    @staticmethod
    def _permutation_impt_selection(data: pd.DataFrame, target: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Feature Selection Using Permutation Importance with Random Forest Model.

        Args:
            data (pd.DataFrame): Input Data
            target (np.array): Target Data

        Returns:
            pd.DataFrame: Feature Selected DataFrame.
        """

        print(f'\n\tRunning Permutation Importance Feature Selection')

        perm_impt_dict = dict()

        model_list = {
            'random_forest': RandomForestClassifier(random_state=0, n_jobs=-1).fit(data, target),
            'gradient_boosting': GradientBoostingClassifier(random_state=0).fit(data, target)
        }

        if 'model_list' in kwargs:
            models = (kwargs['model_list'] if kwargs['model_list']
                      [0].lower() != 'all' else model_list.keys())

            for model_type in models:
                model = model_list[model_type]

                perm_impt = permutation_importance(
                    model, data, target, n_repeats=10, random_state=0, n_jobs=-1)

                perm_df = pd.DataFrame({'feature': data.columns.values, 'avg_impt': perm_impt.importances_mean}).sort_values(
                    by=['avg_impt'], ascending=False).reset_index(drop=True)

                data = data[perm_df[perm_df['avg_impt'] > 0]['feature'].values]
                data['target_label'] = target

                perm_impt_dict[model_type] = data.copy()

        else:
            model = RandomForestClassifier(
                random_state=0, n_jobs=-1).fit(data, target)
            perm_impt = permutation_importance(
                model, data, target, n_repeats=10, random_state=0, n_jobs=-1)

            perm_df = pd.DataFrame({'feature': data.columns.values, 'avg_impt': perm_impt.importances_mean}).sort_values(
                by=['avg_impt'], ascending=False).reset_index(drop=True)

            data = data[perm_df[perm_df['avg_impt'] > 0]['feature'].values]
            data['target_label'] = target

            perm_impt_dict['random_forest'] = data.copy()

        return perm_impt_dict

    @staticmethod
    def _recursive_feature_elimination(data: pd.DataFrame, target: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Selection Feature Using Recursive Feature Elimination with Random Forest Model. Number of features defaults to 25.

        Args:
            data (pd.DataFrame): Input Data
            target (np.array): Target Data

        Returns:
            pd.DataFrame: Feature Selected DataFrame.
        """

        print(f'\n\tRunning RFE Feature Selection')

        rfe_impt_dict = dict()

        model_list = {
            'random_forest': RandomForestClassifier(random_state=0, n_jobs=-1).fit(data, target),
            'gradient_boosting': GradientBoostingClassifier(random_state=0).fit(data, target)
        }

        if 'model_list' in kwargs:
            models = (kwargs['model_list'] if kwargs['model_list']
                      [0].lower() != 'all' else model_list.keys())

            for model_type in models:
                model = model_list[model_type]

                rfe_model = None

                if 'num_feat' in kwargs:
                    rfe_model = RFE(estimator=model, step=1,
                                    n_features_to_select=kwargs['num_feat'], verbose=0).fit(data, target)
                else:
                    rfe_model = RFE(estimator=model, step=1,
                                    n_features_to_select=25, verbose=0).fit(data, target)

                data = data[data.columns[rfe_model.support_]]
                data['target_label'] = target

                rfe_impt_dict[model_type] = data.copy()

        else:
            model = RandomForestClassifier(random_state=0, n_jobs=-1)
            rfe_model = None

            if 'num_feat' in kwargs:
                rfe_model = RFE(estimator=model, step=1,
                                n_features_to_select=kwargs['num_feat'], verbose=0).fit(data, target)
            else:
                rfe_model = RFE(estimator=model, step=1,
                                n_features_to_select=25, verbose=0).fit(data, target)

            data = data[data.columns[rfe_model.support_]]
            data['target_label'] = target

            rfe_impt_dict['random_forest'] = data.copy()

        return rfe_impt_dict

    @staticmethod
    def _model_based_importance(data: pd.DataFrame, target: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Selection Feature Using Random Forest Model's Feature Importance.

        Args:
            data (pd.DataFrame): Input Data
            target (np.array): Target Data

        Returns:
            pd.DataFrame: Feature Selected DataFrame.
        """

        print(f'\n\tRunning Random Forest Feature Selection')

        mbi_feat = dict()

        model_list = {
            'random_forest': RandomForestClassifier(random_state=0, n_jobs=-1).fit(data, target),
        }

        if 'model_list' in kwargs:
            models = (kwargs['model_list'] if kwargs['model_list']
                      [0].lower() != 'all' else model_list.keys())

            for model_type in models:
                model = model_list[model_type]

                impt_df = pd.DataFrame({'feature': data.columns, 'avg_impt': model.feature_importances_}).sort_values(
                    by=['avg_impt'], ascending=False).reset_index(drop=True)

                if 'num_feat' in kwargs:
                    data = data[impt_df.iloc[:kwargs['num_feat']]
                                ['feature'].values]
                else:
                    data = data[impt_df.iloc[:25]['feature'].values]

                data['target_label'] = target

                mbi_feat[model_type] = data.copy()

        else:
            model = RandomForestClassifier(random_state=0, n_jobs=-1)
            impt_df = pd.DataFrame({'feature': data.columns, 'avg_impt': model.feature_importances_}).sort_values(
                by=['avg_impt'], ascending=False).reset_index(drop=True)

            if 'num_feat' in kwargs:
                data = data[impt_df.iloc[:kwargs['num_feat']]
                            ['feature'].values]
            else:
                data = data[impt_df.iloc[:25]['feature'].values]

            data['target_label'] = target

            mbi_feat['random_forest'] = data.copy()

        return mbi_feat

    @staticmethod
    def _regularization_selection(data: pd.DataFrame, target: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Selection Feature Using Lasso Feature Selection.

        Args:
            data (pd.DataFrame): Input Data
            target (np.array): Target Data

        Returns:
            pd.DataFrame: Feature Selected DataFrame.
        """

        print(f'\n\tRunning Lasso Feature Selection')

        reg_feat = dict()

        model_list = {
            'lasso': LogisticRegression(penalty='l1', random_state=0, n_jobs=-1, C=0.1),
            'ridge': RidgeClassifier(alpha=1.0, random_state=0, n_jobs=-1),
            'elasticnet': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=0)
        }

        if 'model_list' in kwargs:
            models = (kwargs['model_list'] if kwargs['model_list']
                      [0].lower() != 'all' else model_list.keys())

            for model_type in models:
                model = model_list[model_type]

                selector = None

                if 'num_feat' in kwargs:
                    selector = SelectFromModel(
                        estimator=model, max_features=kwargs['num_feat'], threshold=-np.inf).fit(data, target)
                else:
                    selector = SelectFromModel(
                        estimator=model, max_features=25, threshold=-np.inf).fit(data, target)

                data = selector.transform(data)
                data['target_label'] = target

                reg_feat[model_type] = data.copy()

        else:
            model = LogisticRegression(
                penalty='l1', random_state=0, n_jobs=-1, solver='saga', max_iter=1000, C=0.1)

            selector = None

            if 'num_feat' in kwargs:
                selector = SelectFromModel(
                    estimator=model, max_features=kwargs['num_feat'], threshold=-np.inf).fit(data, target)
            else:
                selector = SelectFromModel(
                    estimator=model, max_features=25, threshold=-np.inf).fit(data, target)

            data = selector.transform(data)
            data['target_label'] = target

            reg_feat['lasso'] = data.copy()

        return data

    @staticmethod
    def _boruta_selection(data: pd.DataFrame, target: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Selection Feature Using Boruta with Random Forest Model.

        Args:
            data (pd.DataFrame): Input Data
            target (np.array): Target Data

        Returns:
            pd.DataFrame: Feature Selected DataFrame.
        """

        print(f'\n\tRunning Boruta RFC Feature Selection')

        boruta_impt_dict = dict()

        model_list = {
            'random_forest': RandomForestClassifier(random_state=0, n_jobs=-1).fit(data, target),
            'gradient_boosting': GradientBoostingClassifier(random_state=0).fit(data, target)
        }

        if 'model_list' in kwargs:
            models = (kwargs['model_list'] if kwargs['model_list']
                      [0].lower() != 'all' else model_list.keys())

            for model_type in models:
                model = model_list[model_type]

                boruta_model = BorutaPy(
                    estimator=model, n_estimators='auto', verbose=0).fit(data.values, target)

                data = data[data.columns[boruta_model.support_].to_list(
                ) + data.columns[boruta_model.support_weak_].to_list()]
                data['target_label'] = target

                boruta_impt_dict[model_type] = data.copy()

        else:
            model = RandomForestClassifier(random_state=0, n_jobs=-1)
            boruta_model = BorutaPy(
                estimator=model, n_estimators='auto', verbose=0).fit(data.values, target)

            data = data[data.columns[boruta_model.support_].to_list(
            ) + data.columns[boruta_model.support_weak_].to_list()]
            data['target_label'] = target

            boruta_impt_dict['random_forest'] = data.copy()

        return boruta_impt_dict

    # TODO
    @staticmethod
    def _sequencial_forward_selection(data: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
        """Selection Feature Using Sequencial Forward Selection with Random Forest Model.

        Args:
            data (pd.DataFrame): Input Data
            target (np.array): Target Data
            file_name (str): File Path to Save Dataframe

        Returns:
            pd.DataFrame: Feature Selected DataFrame.
        """

        print(f'\n\tRunning Sequencial Forward Selection')

        model = RandomForestClassifier(
            random_state=0, n_jobs=-1).fit(data, target)
        fs_model = sfs(model, k_features=25, verbose=1, forward=True,
                       scoring='roc_auc', cv=5, n_jobs=-1).fit(data, target)

        metrics = fs_model.get_metric_dict()
        cur_max, itr = 0, 0

        for i in range(1, 26):
            try:
                if metrics[i]['avg_score'] > cur_max:
                    cur_max, itr = metrics[i]['avg_score'], i

            except Exception as e:
                print(f'\t\tException Was Raised: {e}')

        selected_feat_fs = list(metrics[itr]['feature_names'])

        data = data[list(selected_feat_fs)]
        data['target_label'] = target

        return data


class FeatureSelection:
    """Transforms Raw Data to Model Injectable Data.

    Methods:
        __init__: Class Initialization Method.
        read_data: Reads Raw Data.
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
            'logit_selection': _Select_Methods._logit_selection,
            'permutation_impt_selection': _Select_Methods._permutation_impt_selection,
            'recursive_feature_elimination': _Select_Methods._recursive_feature_elimination,
            'model_based_importance': _Select_Methods._model_based_importance,
            'regularization_selection': _Select_Methods._regularization_selection,
            'boruta_selection': _Select_Methods._boruta_selection,
            'sequencial_forward_selection': _Select_Methods._sequencial_forward_selection
        }

    def read_data(self, file_type: str, file_path: str, target_feature: str = None, **kwargs) -> tuple:
        """Reads Raw Data

        Args:
            file_path (str): File Path for Raw Data
            file_type (str): File Extension
            target_feature (str, optional): Target Label. Defaults to None.

        Raises:
            ValueError: Invalid File Type.
            TypeError: Target Feature not Specified.

        Returns:
            tuple: Input Data and Target Data
        """

        if file_type not in self._read_data.keys():
            raise ValueError('Invalid File Type.')

        if target_feature is None:
            raise TypeError('Please Specify a Target Feature.')

        try:
            raw_data = self._read_data[file_type](
                file_path=file_path, params=kwargs)

        except Exception as E:
            print(f'File Type Mismatch. {E}')

        input_data = raw_data.drop(columns=[target_feature])
        target_data = raw_data[target_feature].values

        file_name = file_path.split('/')[-1].split('.')[0]

        print('READING FILE -')
        print(
            f'NAME: {file_name}, SHAPE: ROWS - {raw_data.shape[0]}, COLUMNS - {raw_data.shape[1]}, TARGET LABEL: {target_feature}')

        return (input_data, target_data)

    def drop_multicolliner_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Drops Features having High Multicollinearity from the data.

        Args:
            data (pd.DataFrame): Input Data

        Raises:
            ValueError: All Features Dropped.

        Returns:
            pd.DataFrame: Output Data with No Collinear Features
        """

        print('Dropping Multicollinear Features -')

        vif_df = _Utils._calculate_vif(data=data)

        while vif_df['VIF'].max() > 5:
            try:
                id_max = vif_df[['VIF']].idxmax()
                feature_to_drop = str(vif_df.iloc[id_max]['Feature'].values[0])

                print(
                    f'\tFeature: {feature_to_drop}, VIF Value: {round(float(vif_df.iloc[id_max]["VIF"].values[0]), 2)}')

                data = data.drop(columns=[feature_to_drop])

                if data.shape[1] == 0:
                    raise ValueError('All Features Dropped.')

                vif_df = _Utils._calculate_vif(data=data)

            except Exception as e:
                print(f'Error Occured While Calculating VIF - {e}')

            finally:
                break

        print(f'\n\tColumns to keep (After VIF Data): {data.shape[1]}\n')

        return data

    def select_features(self, data: pd.DataFrame, target: np.array, conf: list) -> dict:
        """Runs Feature Selection Pipeline to get the optimal features for Modelling.

        Args:
            data (pd.DataFrame): Input Data.
            target (np.array): Target Data
            conf (list): Configuration for Feature Selection.

        Returns:
            dict: Output Dataset with Selected Features.
        """

        print('Selecting Features -')

        feat_dict = dict()

        for select_method in conf:
            method_name = select_method['select_method']

            if method_name in self._selection_methods.keys():
                feat_dict[method_name] = self._selection_methods[method_name](
                    data.copy(), target, **select_method['params'])

        return feat_dict

    def get_train_test_split_data(self, data_dict: dict, test_size: float = 0.2) -> None:
        print('\nSplitting Data into Train and Test -')

        for data_name, data_value in data_dict.items():
            X = data_value.drop(columns=['target_label'])
            y = data_value['target_label'].values

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=True, stratify=y, random_state=0)

            data_dict[data_name] = {
                'X_train': X_train.copy(), 'X_test': X_test.copy(),
                'y_train': y_train.copy(), 'y_test': y_test.copy()
            }

    def save_data(self, data_dict: object, path: str) -> None:
        """Save the Dataset Pipeline

        Args:
            data_dict (object): Feature Selected Data Dict
            path (str): Saving Path
        """

        print('\nSaving Datasets...')

        joblib.dump({'input_data': data_dict}, path)

    def compile_selection(
        self, file_path: str, file_type: str, target_feature: str, test_size: float, save_path: str, drop_multicolliner_features: bool = True,
        feature_select_conf: list = []
    ) -> dict:
        """Compile and Runs the Feature Selection Pipeline

        Args:
            file_path (str): Raw Data File Path
            file_type (str): File Extention
            target_feature (str): Target Label
            test_size (float): Test Size Percentage
            save_path (str): Path to save Pipeline

        Returns:
            dict: Final Data Dict
        """

        data, target = self.read_data(
            file_path=file_path, file_type=file_type, target_feature=target_feature)

        if drop_multicolliner_features:
            data = self.drop_multicolliner_features(data=data)

        selection_dict = self.select_features(
            data=data, target=target, conf=feature_select_conf)
        selection_dict = self.get_train_test_split_data(
            data_dict=selection_dict, test_size=test_size)

        print('\nFinal Data Info. -')

        for select_type, select_data in selection_dict.items():
            print(
                f'\tSelection Name: {select_type}, SHAPE: ROWS - {select_data["X_train"].shape[0]}, COLUMNS - {select_data["X_train"].shape[1]}')

        self.save_data(data_dict=selection_dict, path=save_path)

        print(f'\nData Transformation Finished\n{"-" * 100}')

        return selection_dict


if __name__ == '__main__':

    # Example Configuration File for Feature Selection.
    config = {
        'file_path': './data/transformed_data_v1.csv',
        'file_type': 'csv',
        'target_feature': 'target_label',
        'drop_multicolliner_features': True,
        'feature_select_conf': [
            {
                'select_method': 'logit_selection',
                'params': {
                    'fit_method': None
                }
            },

            {
                'select_method': 'permutation_impt_selection',
                'params': {
                    'model_list': ['all']
                }
            },

            {
                'select_method': 'recursive_feature_elimination',
                'params': {
                    'model_list': ['all'],
                    'num_feat': 15
                }
            }
        ],
        'test_size': 0.2,
        'save_path': '../data/interim_data/feature_selected_data_v1.joblib'
    }

    feature_selection = FeatureSelection().compile_selection(**config)
