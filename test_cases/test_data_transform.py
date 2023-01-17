import unittest

import pandas as pd
import numpy as np

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


if __name__ == '__main__':
    unittest.main()
