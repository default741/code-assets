import pandas as pd
import numpy as np

import unittest

from src.data_transform import _Read_Data_File, FilePathNotValid, EmptyDataframeObject, _Utils
from src.data_transform import _Transform_Pipeline


class Test_Read_Data_File(unittest.TestCase):

    def setUp(self):
        self.file_path_csv = './data/transformed_data_v1.csv'
        self.file_path_excel = './data/transformed_data_v1.xlsx'
        self.file_path_parquet = './data/transformed_data_v1.parquet'
        self.params = {}
        self.non_existing_file = 'path/to/non_existing_file.csv'

    def test_csv_read(self):
        # Test valid file path
        self.assertIsInstance(_Read_Data_File._read_csv_type(
            self.file_path_csv, self.params), pd.DataFrame)

        # Test invalid file path
        self.assertRaises(FilePathNotValid, _Read_Data_File._read_csv_type,
                          self.non_existing_file, self.params)

    def test_excel_read(self):
        # Test valid file path
        self.assertIsInstance(_Read_Data_File._read_excel_type(
            self.file_path_excel, self.params), pd.DataFrame)

        # Test invalid file path
        self.assertRaises(FilePathNotValid, _Read_Data_File._read_excel_type,
                          self.non_existing_file, self.params)

    def test_parquet_read(self):
        # Test valid file path
        self.assertIsInstance(_Read_Data_File._read_parquet_type(
            self.file_path_parquet, self.params), pd.DataFrame)

        # Test invalid file path
        self.assertRaises(FilePathNotValid, _Read_Data_File._read_parquet_type,
                          self.non_existing_file, self.params)


class TestUtils(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame(
            {'a': [1, 2, 3], 'b': [4, 5, 6], 'c': ['a', 'b', 'c']})

    def test_filter_numeric_columns(self):
        numeric_cols = _Utils._filter_numeric_columns(self.data)
        self.assertIsInstance(numeric_cols, pd.DataFrame)
        self.assertEqual(numeric_cols.shape[1], 2)
        self.assertListEqual(list(numeric_cols.columns), ['a', 'b'])

    def test_filter_categorical_columns(self):
        categorical_cols = _Utils._filter_categorical_columns(self.data)
        self.assertIsInstance(categorical_cols, pd.DataFrame)
        self.assertEqual(categorical_cols.shape[1], 1)
        self.assertListEqual(list(categorical_cols.columns), ['c'])

    def test_filter_numeric_columns_empty_dataframe(self):
        with self.assertRaises(EmptyDataframeObject):
            _Utils._filter_numeric_columns(pd.DataFrame())

    def test_filter_categorical_columns_empty_dataframe(self):
        with self.assertRaises(EmptyDataframeObject):
            _Utils._filter_categorical_columns(pd.DataFrame())


class Test_Drop_Redundent_Columns(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': [
                                 5, 6, 7, 8], 'col3': [9, 10, 11, 12]})
        self.target = np.array([1, 0, 1, 0])
        self.columns_list = ['col1', 'col3']

    def test_init(self):
        # Test for correct input type (list or tuple)
        self.assertRaises(
            TypeError, _Transform_Pipeline._Drop_Redundent_Columns, 'col1')
        self.assertRaises(
            TypeError, _Transform_Pipeline._Drop_Redundent_Columns, {'col1': 'col2'})
        self.assertIsInstance(_Transform_Pipeline._Drop_Redundent_Columns([]),
                              _Transform_Pipeline._Drop_Redundent_Columns)
        self.assertIsInstance(_Transform_Pipeline._Drop_Redundent_Columns(()),
                              _Transform_Pipeline._Drop_Redundent_Columns)

    def test_fit(self):
        # Test that fit returns the correct object
        self.assertIsInstance(_Transform_Pipeline._Drop_Redundent_Columns().fit(
            self.data, self.target), _Transform_Pipeline._Drop_Redundent_Columns)

    def test_transform(self):
        # Test that transform returns a DataFrame
        self.assertIsInstance(_Transform_Pipeline._Drop_Redundent_Columns().fit(
            self.data, self.target).transform(self.data, self.target), pd.DataFrame)

        # Test that specified columns are dropped
        self.assertEqual(list(_Transform_Pipeline._Drop_Redundent_Columns(self.columns_list).fit(
            self.data, self.target).transform(self.data, self.target).columns), ['col2'])

        # Test that error is raised if missing columns are specified in custom_columns_list
        self.assertRaises(TypeError, _Transform_Pipeline._Drop_Redundent_Columns(['col1', 'col4']).fit(
            self.data, self.target).transform, self.data, self.target)

    def test_coverage(self):
        # Test coverage for empty list or tuple
        _Transform_Pipeline._Drop_Redundent_Columns([]).fit(
            self.data, self.target).transform(self.data, self.target)
        _Transform_Pipeline._Drop_Redundent_Columns(()).fit(
            self.data, self.target).transform(self.data, self.target)


class TestDropNullColumns(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({'a': [1, 2, np.nan, 4], 'b': [
                                 5, np.nan, np.nan, 8], 'c': [9, 10, 11, 12]})
        self.target = np.array([1, 2, 3, 4])

    def test_init(self):
        # Test that TypeError is raised when null_percent_threshold is not int or float
        with self.assertRaises(TypeError):
            _Transform_Pipeline._Drop_Null_Columns(
                null_percent_threshold='0.25')

        # Test that ValueError is raised when null_percent_threshold is greater than 1
        with self.assertRaises(ValueError):
            _Transform_Pipeline._Drop_Null_Columns(null_percent_threshold=1.25)

    def test_fit(self):
        # Test that fit method sets drop_cols correctly
        dnc = _Transform_Pipeline._Drop_Null_Columns()
        dnc.fit(self.data, self.target)
        self.assertEqual(dnc.drop_cols, ['b'])

    def test_transform(self):
        # Test that transform method returns correct DataFrame
        dnc = _Transform_Pipeline._Drop_Null_Columns()
        dnc.fit(self.data, self.target)
        result = dnc.transform(self.data, self.target)
        expected = pd.DataFrame({'a': [1, 2, np.nan, 4], 'c': [9, 10, 11, 12]})
        pd.testing.assert_frame_equal(result, expected)

        # Test that transform method returns original DataFrame when drop_cols is empty
        dnc.drop_cols = []
        result = dnc.transform(self.data, self.target)
        expected = self.data
        pd.testing.assert_frame_equal(result, expected)


class TestDropUniqueValueColumns(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [5, 5, 5, 5], 'c': [
                                 9, 10, 11, 12]})
        self.target = np.array([1, 2, 3, 4])

    def test_init(self):
        # Test that TypeError is raised when unique_value_threshold is not int or float
        with self.assertRaises(TypeError):
            _Transform_Pipeline._Drop_Unique_Value_Columns(
                unique_value_threshold='1')

        # Test that ValueError is raised when unique_value_threshold is float and greater than 1
        with self.assertRaises(ValueError):
            _Transform_Pipeline._Drop_Unique_Value_Columns(
                unique_value_threshold=1.5)

    def test_fit(self):
        # Test that fit method sets drop_cols correctly when unique_value_threshold is int
        dnc = _Transform_Pipeline._Drop_Unique_Value_Columns()
        dnc.fit(self.data, self.target)
        self.assertEqual(dnc.drop_cols, ['b'])

        # Test that fit method sets drop_cols correctly when unique_value_threshold is float
        dnc = _Transform_Pipeline._Drop_Unique_Value_Columns(
            unique_value_threshold=0.5)
        dnc.fit(self.data, self.target)
        self.assertEqual(dnc.drop_cols, ['b'])

    def test_transform(self):
        # Test that transform method returns correct DataFrame
        dnc = _Transform_Pipeline._Drop_Unique_Value_Columns()
        dnc.fit(self.data, self.target)
        result = dnc.transform(self.data, self.target)
        expected = pd.DataFrame({'a': [1, 2, 3, 4], 'c': [9, 10, 11, 12]})
        pd.testing.assert_frame_equal(result, expected)

        # Test that transform method returns original DataFrame when drop_cols is empty
        dnc.drop_cols = []
        result = dnc.transform(self.data, self.target)
        expected = self.data
        pd.testing.assert_frame_equal(result, expected)


class Test_Data_Imputation(unittest.TestCase):
    def setUp(self):
        self.X_train = pd.DataFrame(
            {'a': [1, 2, np.nan, 4], 'b': [5, np.nan, np.nan, 8], 'c': [9, 10, 11, 12]})
        self.y_train = np.array([1, 0, 1, 0])
        self.X_test = pd.DataFrame(
            {'a': [np.nan, 2, 3, 4], 'b': [5, np.nan, 7, 8], 'c': [9, np.nan, 11, 12]})
        self.y_test = np.array([1, 0, 1, 0])

    def test_default_init(self):
        imputer = _Transform_Pipeline._Data_Imputation()
        self.assertEqual(imputer.imputation_method, 'iterative')

    def test_simple_init(self):
        imputer = _Transform_Pipeline._Data_Imputation(
            imputation_method='simple')
        self.assertEqual(imputer.imputation_method, 'simple')

    def test_knn_init(self):
        imputer = _Transform_Pipeline._Data_Imputation(imputation_method='knn')
        self.assertEqual(imputer.imputation_method, 'knn')

    def test_invalid_init(self):
        with self.assertRaises(TypeError):
            imputer = _Transform_Pipeline._Data_Imputation(
                imputation_method='invalid')

    def test_fit(self):
        imputer = _Transform_Pipeline._Data_Imputation()
        imputer.fit(self.X_train, self.y_train)
        self.assertIsNotNone(imputer.imputer)
        self.assertIsNotNone(imputer.feature_list)

    def test_transform(self):
        imputer = _Transform_Pipeline._Data_Imputation()
        imputer.fit(self.X_train, self.y_train)
        X_transformed = imputer.transform(self.X_test, self.y_test)
        self.assertIsInstance(X_transformed, pd.DataFrame)
        self.assertEqual(X_transformed.columns.values.tolist(),
                         self.X_train.columns.values.tolist())


class Test_FeatureScaling(unittest.TestCase):
    def test_init(self):
        # Test for invalid scaling method
        with self.assertRaises(TypeError):
            scaler = _Transform_Pipeline._Feature_Scaling(
                scaling_method='invalid')

    def test_fit(self):
        # Create test data
        data = {'feature1': [1, 2, 3], 'feature2': [4, 5, 6]}
        df = pd.DataFrame(data)

        # Test fit
        scaler = _Transform_Pipeline._Feature_Scaling()
        scaler.fit(df)
        self.assertIsNotNone(scaler.scaler)
        self.assertIsNotNone(scaler.feature_list)
        self.assertListEqual(scaler.feature_list, ['feature1', 'feature2'])

    def test_transform(self):
        # Create test data
        data = {'feature1': [1, 2, 3], 'feature2': [4, 5, 6]}
        df = pd.DataFrame(data)

        # Test transform
        scaler = _Transform_Pipeline._Feature_Scaling()
        scaler.fit(df)
        transformed_df = scaler.transform(df)
        self.assertIsInstance(transformed_df, pd.DataFrame)
        self.assertListEqual(list(transformed_df.columns.values), [
                             'feature1', 'feature2'])


class Test_FeatureTransformer(unittest.TestCase):
    def test_init(self):
        # Test for invalid transforming method
        with self.assertRaises(TypeError):
            transformer = _Transform_Pipeline._Feature_Transformer(
                transforming_method='invalid')

    def test_fit(self):
        # Create test data
        data = {'feature1': [1, 2, 3], 'feature2': [4, 5, 6]}
        df = pd.DataFrame(data)

        # Test fit
        transformer = _Transform_Pipeline._Feature_Transformer()
        transformer.fit(df)
        self.assertIsNotNone(transformer.transformer)
        self.assertIsNotNone(transformer.feature_list)
        self.assertListEqual(transformer.feature_list,
                             ['feature1', 'feature2'])

    def test_transform(self):
        # Create test data
        data = {'feature1': [1, 2, 3], 'feature2': [4, 5, 6]}
        df = pd.DataFrame(data)

        # Test transform
        transformer = _Transform_Pipeline._Feature_Transformer()
        transformer.fit(df)
        transformed_df = transformer.transform(df)
        self.assertIsInstance(transformed_df, pd.DataFrame)
        self.assertListEqual(list(transformed_df.columns.values), [
                             'feature1', 'feature2'])


def run_test():
    testing_list = list()
    testing_classes = [
        Test_Read_Data_File, TestUtils, Test_Drop_Redundent_Columns, TestDropNullColumns, TestDropUniqueValueColumns,
        Test_Data_Imputation, Test_FeatureScaling, Test_FeatureTransformer
    ]

    for test_class in testing_classes:
        testing_list.append(unittest.defaultTestLoader.loadTestsFromTestCase(
            test_class))

    suite = unittest.TestSuite(testing_list)

    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':
    unittest.main()
