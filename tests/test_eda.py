import unittest
import pandas as pd
from scripts.eda import EDA  # Adjust the import based on your file structure

class TestEDA(unittest.TestCase):
    def setUp(self):
        # Sample DataFrame for testing
        data = {
            'A': [1, 2, 3, 4, 5],
            'B': [5, 4, 3, 2, 1],
            'C': ['a', 'b', 'a', 'b', 'a']
        }
        self.df = pd.DataFrame(data)
        self.eda = EDA(self.df)

    def test_overview(self):
        expected_shape = (5, 3)
        expected_columns = ['A', 'B', 'C']
        overview = self.eda.overview()
        self.assertEqual(overview['shape'], expected_shape)
        self.assertListEqual(overview['columns'], expected_columns)

    def test_summary_statistics(self):
        summary = self.eda.summary_statistics()
        self.assertEqual(summary.shape[0], 8)  # Should return 8 summary statistics

    def test_data_types(self):
        data_types = self.eda.data_types()
        expected_types = pd.Series({'A': 'int64', 'B': 'int64', 'C': 'object'})
        pd.testing.assert_series_equal(data_types, expected_types)

    def test_check_missing_values(self):
        missing_values = self.eda.check_missing_values()
        self.assertEqual(len(missing_values), 0)  # No missing values in the sample data

    def test_plot_distribution(self):
        # This test would typically check if the method runs without error
        try:
            self.eda.plot_distribution('A')
        except Exception as e:
            self.fail(f"plot_distribution raised {type(e).__name__} unexpectedly!")

    def test_plot_correlation_matrix(self):
        # This test would typically check if the method runs without error
        try:
            self.eda.plot_correlation_matrix()
        except Exception as e:
            self.fail(f"plot_correlation_matrix raised {type(e).__name__} unexpectedly!")

    def test_detect_outliers(self):
        # This test would typically check if the method runs without error
        try:
            self.eda.detect_outliers('A')
        except Exception as e:
            self.fail(f"detect_outliers raised {type(e).__name__} unexpectedly!")

if __name__ == '__main__':
    unittest.main()
