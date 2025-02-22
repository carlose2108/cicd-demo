import unittest
import pandas as pd
import os
import shutil

from app.model import Model
from unittest.mock import patch

class TestModel(unittest.TestCase):

    def setUp(self):
        """Setup test environment with a sample DataFrame."""
        self.sample_data = pd.DataFrame({
            "invoiceId": [1, 2, 3, 4, 5],
            "payerId": [101, 102, 103, 104, 105],
            "overdueDays": [10, 40, 20, 50, 5],
            "amount": [100.5, 200.0, 150.75, 300.0, 250.0],
        })
        # self.model = Model(self.sample_data)
        self.model_class = Model(self.sample_data)

    def test_init_valid_dataframe(self):
        """Test Model initialization with a valid DataFrame."""
        self.assertIsInstance(self.model_class.df, pd.DataFrame)

    def test_init_invalid_dataframe(self):
        """Test that an invalid DataFrame raises an error."""
        with self.assertRaises(ValueError):
            Model("invalid_data")

    def test_prepare_data(self):
        """Test prepare_data method applies StandardScaler and creates invoice_risk."""
        processed_df = self.model_class.prepare_data()
        self.assertIn("invoice_risk", processed_df.columns)
        self.assertEqual(processed_df["invoice_risk"].sum(), 2)
        self.assertAlmostEqual(processed_df["amount"].mean(), 0, delta=0.1)

    def test_prepare_data_missing_column(self):
        """Test prepare_data raises error if 'overdueDays' column is missing."""
        sample_data_invalid = self.sample_data.drop(columns=["overdueDays"])
        model_invalid = Model(sample_data_invalid)
        processed_df = model_invalid.prepare_data()
        self.assertIsNone(processed_df)

    def test_impute_missing(self):
        """Test impute_missing does not modify DataFrame if no missing values."""
        df_no_missing = self.model_class.impute_missing()
        self.assertFalse(df_no_missing.isnull().values.any())  # No missing values

    def test_fit_model_trains(self):
        """Test fit method trains a model and assigns it to self.model."""
        self.model_class.fit()
        self.assertIsNotNone(self.model_class.model)
        self.assertEqual(self.model_class.model.__class__.__name__, "XGBClassifier")

    def test_model_summary(self):
        """Test model_summary method returns correct summary."""
        # Before training
        self.assertEqual(self.model_class.model_summary(), "Model has not been trained yet.")
        # After training
        self.model_class.fit()
        self.assertIn("Trained XGBClassifier", self.model_class.model_summary())

    def test_predict_valid(self):
        """Test predict method returns expected output after training."""
        self.model_class.fit()
        predictions = self.model_class.predict([1, 2])
        self.assertIsInstance(predictions, pd.Series)
        self.assertEqual(predictions.name, "predicted_invoice_risk")

    def test_save_trained_model(self):
        """Test save method correctly saves the model."""
        self.model_class.fit()
        save_path = "test_model_dir"
        model_name = "test_model"
        self.model_class.save(save_path, model_name)
        # Check if the path is created
        self.assertTrue(os.path.exists(save_path))
        # Remove file
        shutil.rmtree(save_path)
        self.assertFalse(os.path.exists(save_path))

    def test_save_untrained_model(self):
        """Test save method does nothing if model is not trained."""
        with self.assertLogs() as cap:
            self.model_class.save("dummy_path", "model")

        expected_log = "Model has not been trained yet."
        self.assertEqual(expected_log, cap.records[0].getMessage())


    def test_preprocess_data_api(self):
        """Test preprocess_data_api correctly scales input data."""
        invoice_ids = [1, 2]
        preprocessed_data = self.model_class.preprocess_data_api(self.sample_data, invoice_ids)
        self.assertEqual(preprocessed_data.shape[0], 2)
        self.assertAlmostEqual(preprocessed_data.mean(), 0, delta=0.1)

    def test_preprocess_data_api_missing_column(self):
        """Test preprocess_data_api raises error if 'invoiceId' column is missing."""
        sample_data_invalid = self.sample_data.drop(columns=["invoiceId"])
        result = self.model_class.preprocess_data_api(sample_data_invalid, [1])
        self.assertIsNone(result)
