import unittest
from test_cases.test_data_transform import Test_Read_Data_File, TestUtils, Test_Drop_Redundent_Columns

if __name__ == '__main__':
    test_read_data_file = unittest.defaultTestLoader.loadTestsFromTestCase(
        Test_Read_Data_File)
    test_utils = unittest.defaultTestLoader.loadTestsFromTestCase(TestUtils)
    test_tp_drop_redundent_cols = unittest.defaultTestLoader.loadTestsFromTestCase(
        Test_Drop_Redundent_Columns)

    suite = unittest.TestSuite(
        [test_read_data_file, test_utils, test_tp_drop_redundent_cols])

    unittest.TextTestRunner(verbosity=2).run(suite)
