# Custom Data Transformation Class Asset to be Used for any Future Projects - v0.2

# Author: Abdemanaaf Ghadiali
# Copyright: Copyright 2023, Data_Transform, https://data-transform.com
# Version: 0.2
# Status: Development
# Code Style: PEP8 Style Guide
# MyPy Status: NA (Not Tested)


from sklearn.base import TransformerMixin, BaseEstimator  # type: ignore
from sklearn.experimental import enable_iterative_imputer  # type: ignore
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer  # type: ignore
from sklearn.preprocessing import MinMaxScaler, RobustScaler, MaxAbsScaler, StandardScaler  # type: ignore
from sklearn.preprocessing import PowerTransformer, QuantileTransformer  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.ensemble import IsolationForest  # type: ignore

from typing import Union

import joblib
import os

import pandas as pd
import numpy as np


class FilePathNotValid(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class EmptyDataframeObject(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class _Read_Data_File:
    """The purpose of this class is to provide a protected way to read data from different file types such as CSV, EXCEL (xlsx), and Parquet.

    How to Use:
        data_file = _Read_Data_File()
        params = {'sep': ',', 'header': 0, 'index_col': 0}
        df = data_file._read_csv_type('path/to/file.csv', params)

    Requirements:
        This class requires the 'pandas' library to be imported. The class uses the functions provided by pandas to read the file data.

    Methods:
        _read_csv_type: Reads CSV file type
        _read_excel_type: Reads XLSX file type
        _read_parquet_type: Reads PARQUET file type
    """

    @staticmethod
    def _read_csv_type(file_path: str, params: dict) -> pd.DataFrame:
        """Used to read and import data from a CSV (Comma Separated Values) file. This function takes the file path of the CSV file as its
        input and returns a DataFrame object.

        Args:
            file_path (str): The path refers to the location of the file on a computer or network, in order to specify where the file should be read from.
            params (dict): Dictionary that contains extra parameters to pass to the underlying pandas function that is used to read the file.

        Raises:
            FilePathNotValid: The specified file path is not valid.

        Returns:
            pd.DataFrame: A raw data file that contains unprocessed, original data in its native format, without any manipulation or modification.
        """

        # Checks if the file path is valid or not.
        if not (os.path.exists(file_path) and os.path.isfile(file_path)):
            raise FilePathNotValid(f'{file_path} is not a valid file path.')

        return pd.read_csv(file_path, **params)

    @staticmethod
    def _read_excel_type(file_path: str, params: dict) -> pd.DataFrame:
        """Used to read and import data from a EXCEL file. This function takes the file path of the EXCEL file as its
        input and returns a DataFrame object.

        Args:
            file_path (str): The path refers to the location of the file on a computer or network, in order to specify where the file should be read from.
            params (dict): Dictionary that contains extra parameters to pass to the underlying pandas function that is used to read the file.

        Raises:
            FilePathNotValid: The specified file path is not valid.

        Returns:
            pd.DataFrame: A raw data file that contains unprocessed, original data in its native format, without any manipulation or modification.
        """

        # Checks if the file path is valid or not.
        if not (os.path.exists(file_path) and os.path.isfile(file_path)):
            raise FilePathNotValid(f'{file_path} is not a valid file path.')

        return pd.read_excel(file_path, **params)

    @staticmethod
    def _read_parquet_type(file_path: str, params: dict) -> pd.DataFrame:
        """Used to read and import data from a Parquet file. This function takes the file path of the Parquet file as its
        input and returns a DataFrame object.

        Args:
            file_path (str): The path refers to the location of the file on a computer or network, in order to specify where the file should be read from.
            params (dict): Dictionary that contains extra parameters to pass to the underlying pandas function that is used to read the file.

        Raises:
            FilePathNotValid: The specified file path is not valid.

        Returns:
            pd.DataFrame: A raw data file that contains unprocessed, original data in its native format, without any manipulation or modification.
        """

        # Checks if the file path is valid or not.
        if not (os.path.exists(file_path) and os.path.isfile(file_path)):
            raise FilePathNotValid(f'{file_path} is not a valid file path.')

        return pd.read_parquet(file_path, **params)


class _Utils:
    """Utility Class for Helper Functions: These methods are intended to be used as helper functions inside the class to make it
    more readable and maintainable. The underscore prefix indicates that it is intended for internal use rather than for external use.

    How to Use:
        data_file = _Utils()
        df = _Utils._filter_numeric_columns(dataframe_object)

    Requirements:
        This class requires the 'pandas' library to be imported. The class uses the functions provided by pandas to filter features based on their type.

    Methods:
        _filter_numeric_columns: This method is used to filter the numeric type columns from a given dataset.
        _filter_categorical_columns: This method is used to filter the categorical type columns from a given dataset.
    """

    @staticmethod
    def _filter_numeric_columns(data: pd.DataFrame) -> pd.DataFrame:
        """This method is used to filter the numeric type columns from a given dataset.

        Args:
            data (pd.DataFrame): A raw data file that contains unprocessed, original data in its native format, without any manipulation or modification.

        Raises:
            EmptyDataframeObject: Raises an exception if the Dataframe passed is empty. This is to ensure that the method is not applied on an empty dataset.

        Returns:
            pd.DataFrame: Returns a Dataframe containing only the numeric type columns of the input dataset.
        """

        # Checks if the dataframe object is not empty.
        if data.empty:
            raise EmptyDataframeObject('Dataframe is Empty.')

        return data.select_dtypes(include=['float64', 'int64', 'uint8'])

    @staticmethod
    def _filter_categorical_columns(data: pd.DataFrame) -> pd.DataFrame:
        """This method is used to filter the categorical type columns from a given dataset.

        Args:
            data (pd.DataFrame): A raw data file that contains unprocessed, original data in its native format, without any manipulation or modification.

        Raises:
            EmptyDataframeObject: Raises an exception if the Dataframe passed is empty. This is to ensure that the method is not applied on an empty dataset.

        Returns:
            pd.DataFrame: returns a Dataframe containing only the object type columns of the input dataset.
        """

        # Checks if the dataframe object is not empty.
        if data.empty:
            raise EmptyDataframeObject('Dataframe is Empty.')

        return data.select_dtypes(include='object')


class _Transform_Pipeline:

    class _Drop_Redundent_Columns(BaseEstimator, TransformerMixin):
        """Custom Pipeline Class to Drop Redundent Features List from the User.

        Methods:
            __init__: Class Initialization Method.
            fit: Fit the class wrt. the given data.
            transform: Transform the Data wrt. the fitted data.
        """

        def __init__(self, custom_columns_list: Union[list, tuple] = []) -> None:
            """Class Initialization Method.

            Args:
                custom_columns_list (Union[list, tuple], optional): Custom List of Features given by the User. Defaults to [].

            Raises:
                TypeError: _description_
            """

            if not isinstance(custom_columns_list, (list, tuple)):
                raise TypeError(
                    'Columns List should be either a List or a Tuple.')

            self.custom_columns_list: list = custom_columns_list

        def fit(self, X: pd.DataFrame, y: np.ndarray = np.ndarray(shape=0)) -> object:
            """Fit the class wrt. the given data.

            Args:
                X (pd.DataFrame): Input Data
                y (np.array): Target Data
            """
            return self

        def transform(self, X: pd.DataFrame, y: np.ndarray = np.ndarray(shape=0)) -> pd.DataFrame:
            """Transform the Data wrt. the fitted data.

            Args:
                X (pd.DataFrame): Input Data
                y (np.array): Target Data

            Raises:
                TypeError: Missing Features in DataFrame

            Returns:
                pd.DataFrame: Final Dataframe after transformation step
            """
            if len(self.custom_columns_list) == 0:
                return X.reset_index(drop=True)

            if not all(feat in list(X.columns) for feat in self.custom_columns_list):
                raise TypeError(
                    'Missing Features from Dataframe specified in Custom List.')

            return X.drop(columns=self.custom_columns_list).reset_index(drop=True)

    class _Drop_Null_Columns(BaseEstimator, TransformerMixin):
        """Custom Pipeline Class to Drop Redundent Features having all null Values.

        Methods:
            __init__: Class Initialization Method.
            fit: Fit the class wrt. the given data.
            transform: Transform the Data wrt. the fitted data.
        """

        def __init__(self, null_percent_threshold: Union[float, int] = 0.25) -> None:
            """Class Initialization Method.

            Args:
                null_percent_threshold (float, optional): Threshold above which nulls are to be dropped. Defaults to 0.25.

            Raises:
                TypeError: If null_percent_threshold is not an int or float
                ValueError: If null_percent_threshold is > than 1
            """

            if not isinstance(null_percent_threshold, (int, float)):
                raise TypeError(
                    'Input Parameter should be an Integer and Float in Drop Null Columns.')

            if null_percent_threshold > 1:
                raise ValueError(
                    'Null Percent Threshold cannot be greater than 1')

            self.null_percent_threshold = null_percent_threshold
            self.drop_cols = None

        def fit(self, X: pd.DataFrame, y: np.ndarray = np.ndarray(shape=0)) -> object:
            """Fit the class wrt. the given data.

            Args:
                X (pd.DataFrame): Input Data
                y (np.ndarray): Target Data. Defaults to np.ndarray(shape=0)
            """

            self.drop_cols = [i for i in X.columns if (
                X[i].isna().mean() >= self.null_percent_threshold)]

            return self

        def transform(self, X: pd.DataFrame, y: np.ndarray = np.ndarray(shape=0)) -> pd.DataFrame:
            """Transform the Data wrt. the fitted data.

            Args:
                X (pd.DataFrame): Input Data
                y (np.ndarray): Target Data. Defaults to np.ndarray(shape=0)
            """

            if len(self.drop_cols) == 0:
                return X.reset_index(drop=True)

            return X.drop(columns=self.drop_cols).reset_index(drop=True)

    class _Drop_Unique_Value_Columns(BaseEstimator, TransformerMixin):
        """Custom Pipeline Class to Drop Redundent Features having only one Unique Value.

        Methods:
            __init__: Class Initialization Method.
            fit: Fit the class wrt. the given data.
            transform: Transform the Data wrt. the fitted data.
        """

        def __init__(self, unique_value_threshold: Union[float, int] = 1) -> None:
            """Class Initialization Method.

            Args:
                custom_columns_list (list): Custom List of Features given by the User. Defaults to 1.

            Raises:
                TypeError: If unique_value_threshold is not an int or float
                ValueError: If unique_value_threshold is a float and > than 1
            """

            if not isinstance(unique_value_threshold, (int, float)):
                raise TypeError(
                    'Input Parameter should be an Integer and Float in Drop Unique Value Columns.')

            if isinstance(unique_value_threshold, float) and unique_value_threshold > 1:
                raise ValueError(
                    'If unique_value_threshold is a float, it cannot be greater than 1.')

            self.unique_value_threshold = unique_value_threshold
            self.drop_cols = None

        def fit(self, X: pd.DataFrame, y: np.ndarray = np.ndarray(shape=0)) -> object:
            """Fit the class wrt. the given data.

            Args:
                X (pd.DataFrame): Input Data
                y (np.ndarray): Target Data. Defaults to np.ndarray(shape=0)
            """

            if isinstance(self.unique_value_threshold, int):
                self.drop_cols = [i for i in X.columns if (
                    X[i].dropna().nunique() <= self.unique_value_threshold)]
            else:
                self.drop_cols = [i for i in X.columns if (
                    X[i].dropna().nunique() / X[i].count() <= self.unique_value_threshold)]

            return self

        def transform(self, X: pd.DataFrame, y: np.ndarray = np.ndarray(shape=0)) -> pd.DataFrame:
            """Transform the Data wrt. the fitted data.

            Args:
                X (pd.DataFrame): Input Data
                y (np.ndarray): Target Data. Defaults to np.ndarray(shape=0)
            """

            if len(self.drop_cols) == 0:
                return X.reset_index(drop=True)

            return X.drop(columns=self.drop_cols).reset_index(drop=True)

    class _Data_Imputation(BaseEstimator, TransformerMixin):
        """Imputes Missing/NaN Values of Individual Feature with median of the Column Values.

        Methods:
            __init__: Class Initialization Method.
            fit: Fit the class wrt. the given data.
            transform: Transform the Data wrt. the fitted data.
        """

        def __init__(self, imputation_method: str = 'iterative', **kwargs) -> None:
            """Class Initialization Method.

            Args:
                imputation_method (str, optional): Which method to use for Imputation. Defaults to 'iterative'.

            Raises:
                TypeError: Invalid Method Specified
            """

            self.IMPUTATION_LIST = {
                'iterative': IterativeImputer(**kwargs),
                'simple': SimpleImputer(**kwargs),
                'knn': KNNImputer(**kwargs)
            }

            if not imputation_method in self.IMPUTATION_LIST:
                raise TypeError('Invalid Imputation Method.')

            self.imputation_method = imputation_method

            self.imputer = None
            self.feature_list = None

        def fit(self, X: pd.DataFrame, y: np.ndarray = np.ndarray(shape=0)):
            """Fit the class wrt. the given data.

            Args:
                X (pd.DataFrame): Input Data
                y (np.ndarray): Target Data. Defaults to np.ndarray(shape=0)
            """

            self.feature_list = X.columns.values

            self.imputer = self.IMPUTATION_LIST[self.imputation_method]
            self.imputer.fit(X)

            return self

        def transform(self, X: pd.DataFrame, y: np.ndarray = np.ndarray(shape=0)):
            """Transform the Data wrt. the fitted data.

            Args:
                X (pd.DataFrame): Input Data
                y (np.ndarray): Target Data. Defaults to np.ndarray(shape=0)
            """
            return pd.DataFrame(data=self.imputer.transform(X), columns=self.feature_list).reset_index(drop=True)

    class _Feature_Scaling(BaseEstimator, TransformerMixin):
        """Standardizes Data to One Common Scale.

        Methods:
            __init__: Class Initialization Method.
            fit: Fit the class wrt. the given data.
            transform: Transform the Data wrt. the fitted data.
        """

        def __init__(self, scaling_method: str = 'standard', **kwargs) -> None:
            """Class Initialization Method.

            Args:
                scaling_method (str, optional): Which method to use for Scaling. Defaults to 'standard'.

            Raises:
                TypeError: Invalid Method Specified
            """

            self.SCALING_LIST = {
                'standard': StandardScaler(**kwargs),
                'robust': RobustScaler(**kwargs),
                'minmax': MinMaxScaler(**kwargs),
                'maxabs': MaxAbsScaler(**kwargs)
            }

            if not scaling_method in self.SCALING_LIST:
                raise TypeError('Invalid Scaling Method.')

            self.scaling_method = scaling_method

            self.scaler = None
            self.feature_list = None

        def fit(self, X: pd.DataFrame, y: np.ndarray = np.ndarray(shape=0)):
            """Fit the class wrt. the given data.

            Args:
                X (pd.DataFrame): Input Data
                y (np.ndarray): Target Data. Defaults to np.ndarray(shape=0)
            """

            self.feature_list = X.columns.values

            self.scaler = self.SCALING_LIST[self.scaling_method]
            self.scaler.fit(X)

            return self

        def transform(self, X: pd.DataFrame, y: np.ndarray = np.ndarray(shape=0)):
            """Transform the Data wrt. the fitted data.

            Args:
                X (pd.DataFrame): Input Data
                y (np.ndarray): Target Data. Defaults to np.ndarray(shape=0)
            """
            return pd.DataFrame(data=self.scaler.transform(X), columns=self.feature_list).reset_index(drop=True)

    class _Feature_Transformer(BaseEstimator, TransformerMixin):
        """Transforms Data using Log Transformation.

        Methods:
            __init__: Class Initialization Method.
            fit: Fit the class wrt. the given data.
            transform: Transform the Data wrt. the fitted data.
        """

        def __init__(self, transforming_method: str = 'power', **kwargs) -> None:
            """Class Initialization Method.

            Args:
                transforming_method (str, optional): Which method to use for Transforming. Defaults to 'standard'.

            Raises:
                TypeError: Invalid Method Specified
            """

            self.TRANSFORMING_LIST = {
                'power': PowerTransformer(**kwargs),
                'quantile': QuantileTransformer(**kwargs)
            }

            if not transforming_method in self.TRANSFORMING_LIST:
                raise TypeError('Invalid Transforming Method.')

            self.transforming_method = transforming_method

            self.transformer = None
            self.feature_list = None

        def fit(self, X: pd.DataFrame, y: np.ndarray = np.ndarray(shape=0)):
            """Fit the class wrt. the given data.

            Args:
                X (pd.DataFrame): Input Data
                y (np.ndarray): Target Data. Defaults to np.ndarray(shape=0)
            """

            self.feature_list = X.columns.values

            self.transformer = self.TRANSFORMING_LIST[self.transforming_method]
            self.transformer.fit(X)

            return self

        def transform(self, X: pd.DataFrame, y: np.array = None):
            """Transform the Data wrt. the fitted data.

            Args:
                X (pd.DataFrame): Input Data
                y (np.array): Target Data
            """
            return pd.DataFrame(data=self.transformer.transform(X), columns=self.feature_list).reset_index(drop=True)


class DataTransform:
    """Transforms Raw Data to Model Injectable Data.

    Methods:
        __init__: Class Initialization Method.
        read_data: Reads Raw Data.
        encode_features: Converts Categorical Features to One-Hot Encoded Features.
        transdorm: Runs Transformations Pipeline.
        remove_outliers: Removes Outliers using Isolation Forest.
        save_pipeline: Saves the Transformation Pipeline.
        run: Runs all the Methods Sequencially.
    """

    def __init__(self) -> None:
        """Class Initialization Method.
        """

        self._read_data = {
            'csv': _Read_Data_File._read_csv_type,
            'xlsx': _Read_Data_File._read_excel_type,
            'parquet': _Read_Data_File._read_parquet_type
        }

        self._transform_steps = {
            'drop_redundent_columns': _Transform_Pipeline._Drop_Redundent_Columns,
            'drop_null_columns': _Transform_Pipeline._Drop_Null_Columns,
            'drop_unique_value_columns': _Transform_Pipeline._Drop_Unique_Value_Columns,
            'data_imputation': _Transform_Pipeline._Data_Imputation,
            'feature_scaling': _Transform_Pipeline._Data_Imputation,
            'feature_transformer': _Transform_Pipeline._Feature_Transformer
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

    def encode_features(self, feature_list: list, data: pd.DataFrame) -> pd.DataFrame:
        """Encodes Categorical Data to One-Hot Encoded Data.

        Args:
            feature_list (list): Categorical Features
            data (pd.DataFrame): Input Data

        Returns:
            pd.DataFrame: Encoded Data
        """

        print('CREATING DUMMY FEATURES -')

        if feature_list is None:
            feature_list = _Utils._filter_categorical_columns(data=data)

        for feature in feature_list:
            if feature not in data.columns:
                raise ValueError('Feature not present in Dataset.')

            data = pd.concat([data, pd.get_dummies(
                data[feature], prefix=feature.lower(), drop_first=True)], axis=1)

            print(
                f'FEATURE NAME: {feature}, UNIQUE VALUES: {data[feature].nunique()}')

            data = data.drop(columns=[feature]).reset_index(drop=True)

        return data

    def run_transform(self, data: pd.DataFrame, transform_conf: dict) -> pd.DataFrame:
        """Creates and Runs Transformation

        Args:
            data (pd.DataFrame): Input Data
            transform_conf (dict): Transformation Configuration

        Returns:
            pd.DataFrame: Transformed Data
        """

        print('Running Transformation Pipeline -')
        data = _Utils._filter_numeric_columns(data=data)

        for method in transform_conf:
            if method['method_name'] not in self._transform_steps.keys():
                raise ValueError('Method Name Not Found.')

        pipeline_steps = [(method['method_name'], self._transform_steps[method['method_name']](
            **method['params'])) for method in transform_conf]

        pipeline = Pipeline(steps=pipeline_steps, verbose=True)
        data = pipeline.fit_transform(data, None)

        # Rounding Dataframe so the Model is Not Sensitive to miniscule changes.
        data = data.round(5)

        print('Transformation Pipeline Excecution Finished')

        return (data, pipeline)

    def remove_outliers(self, data: pd.DataFrame, target: np.array, contamination_factor: float = 0.01) -> tuple:
        """Removes Outliers Based on Isolation Forest.

        Args:
            data (pd.DataFrame): Transformed Data
            target (np.array): Target Data
            contamination_factor (float, optional): Isolation Forest Parameter. Defaults to 0.01.

        Returns:
            tuple: Outlier Free Data
        """

        print('Dropping Outliers -')

        outlier_model = IsolationForest(
            random_state=0, verbose=0, n_jobs=-1, contamination=contamination_factor)
        outlier_model.fit(data)

        index_filter = np.where(outlier_model.predict(data) == 1, True, False)

        print(
            f'Number of Outliers to be Dropped: {data.shape[0] - sum(index_filter)}')

        data = data.filter(items=data.index[index_filter], axis=0)
        target = target[index_filter]

        return (data, target)

    def save_pipeline(self, pipeline: object, path: str, feature_list: list) -> None:
        """Save the Transformation Pipeline

        Args:
            pipeline (object): Pipeline object
            path (str): Saving Path
            feature_list (list): Original Data Feature List
        """

        print('Saving Pipeline...')

        transform_pipeline_dict = {
            'pipeline': pipeline, 'custom_functions': _Transform_Pipeline, 'feature_list': feature_list}
        joblib.dump(transform_pipeline_dict, path)

        print(f'Pipeline Saved to path: {path}')

    def compile_pipeline(
        self, file_path: str, file_type: str, target_feature: str, feature_list: list, transform_conf: dict,
        save_path: str, remove_outlier: bool = True, contamination_factor: float = None
    ) -> pd.DataFrame:
        """Compile and Runs the Data Transformation Pipeline

        Args:
            file_path (str): Raw Data File Path
            file_type (str): File Extention
            target_feature (str): Target Label
            feature_list (list): Categorical Feature List
            conf (dict): Transformation Configuration
            save_path (str): Path to save Pipeline
            contamination_factor (float): Isolation Forest Contamination Factor

        Returns:
            pd.DataFrame: Final Data
        """

        print('Data Transformation Started')

        data, target = self.read_data(
            file_path=file_path, file_type=file_type, target_feature=target_feature)
        data = self.encode_features(data=data, feature_list=feature_list)

        data, pipeline = self.run_transform(
            data=data, transform_conf=transform_conf)

        if remove_outlier:
            data, target = self.remove_outliers(
                data=data, target=target, contamination_factor=contamination_factor)

        self.save_pipeline(pipeline=pipeline, path=save_path,
                           feature_list=data.columns.values)

        data['target_label'] = target

        print('Final Data Info. -')
        print(
            f'SHAPE: ROWS - {data.shape[0]}, COLUMNS - {data.shape[1]}')
        print(f'Data Transformation Finished')

        return data


if __name__ == '__main__':

    # Example Configuration File for Data Transformation.
    config = {
        'file_path': './data/paint_quality_assurance_data.csv',
        'file_type': 'csv',
        'target_feature': 'Dust Defect',
        'feature_list': ['Platform', 'Primer Color', 'Topcoat Color'],
        'transform_conf': [
            {
                'method_name': 'drop_redundent_columns',
                'params': {}
            },
            {
                'method_name': 'drop_null_columns',
                'params': {
                    'null_percent_threshold': 0.2
                }
            },
            {
                'method_name': 'drop_unique_value_columns',
                'params': {}
            },
            {
                'method_name': 'data_imputation',
                'params': {}
            },
            {
                'method_name': 'feature_scaling',
                'params': {}
            },
            {
                'method_name': 'feature_transformer',
                'params': {}
            }
        ],
        'remove_outlier': True,
        'contamination_factor': 0.01,
        'save_path': './data/transform_pipeline.joblib'
    }

    transformed_data = DataTransform().compile_pipeline(**config)
    transformed_data.to_csv(
        './data/transformed_data_v1.csv', index=False)
