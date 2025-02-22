import os
import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from xgboost import XGBClassifier

from typing import List
import logging

logger = logging.getLogger(__name__)


class Model:
    def __init__(self, sample_df: pd.DataFrame):
        """Initialize the class."""
        if not isinstance(sample_df, pd.DataFrame):
            raise ValueError("Pandas DataFrame is not provided. You must pass a valid DataFrame.")

        self.df = sample_df
        self.model = None
        self.scaler = StandardScaler()


    def prepare_data(self, df: pd.DataFrame = None) -> pd.DataFrame | None:
        """Prepare data."""
        try:
            if df is None:
                df = self.df

            if "overdueDays" not in df.columns:
                raise ValueError("Column 'overdueDays' is missing in the DataFrame.")

            df["invoice_risk"] = np.where(df["overdueDays"] > 30, 1, 0)

            excluded_cols = ["invoiceId", "payerId", "invoice_risk"]
            feature_cols = df.columns.difference(excluded_cols)

            df[feature_cols] = self.scaler.fit_transform(df[feature_cols])

            return df

        except Exception as e:
            logger.error(f"Error in prepare_data: {e}")
            return None

    def impute_missing(self, df: pd.DataFrame = None) -> pd.DataFrame | None:
        """Fill missing values."""
        try:
            if df is None:
                df = self.df

            if df.isnull().any().any():
                logger.info("Warning: DataFrame has missing values. Handling missing values accordingly.")

            return df

        except Exception as e:
            logger.error(f"Error in impute_missing: {e}")
            return None

    def fit(self, model=None) -> None:
        """Fit the model."""
        try:
            if model is None:
                model = XGBClassifier()

            df = self.prepare_data()
            if df is None:
                raise ValueError("Failed to prepare data.")

            df = self.impute_missing(df)
            if df is None:
                raise ValueError("Failed to impute missing values.")

            X = df.drop(columns=["invoiceId", "invoice_risk"])
            y = df["invoice_risk"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=42)
            model.fit(X_train, y_train)

            self.model = model
            y_pred = model.predict(X_test)

            logger.info("Classification Report:\n", classification_report(y_test, y_pred))

        except Exception as e:
            logger.error(f"Error in fit: {e}")

    def model_summary(self) -> str | None:
        """Create a short summary of the model."""
        try:
            if self.model is not None:
                return f"Trained {type(self.model).__name__} on invoice risk prediction."
            else:
                return "Model has not been trained yet."
        except Exception as e:
            logger.error(f"Error in model_summary: {e}")
            return None

    def predict(self, invoice_ids: List[int]) -> pd.Series:
        """Make a set of predictions."""
        try:
            if self.model is None:
                raise ValueError("Model has not been trained yet. Use the fit method first.")

            invoices_to_predict = self.df[self.df["invoiceId"].isin(invoice_ids)].drop(
                columns=["invoiceId", "invoice_risk"])

            predictions = self.model.predict(invoices_to_predict)

            return pd.Series(predictions, name="predicted_invoice_risk")

        except Exception as e:
            logger.error(f"Error in predict: {e}")
            return pd.Series()

    def save(self, path: str, model_name: str) -> None:
        """Save the model."""
        try:
            if self.model is not None:
                if not os.path.exists(path):
                    os.makedirs(os.path.abspath(path))

                pickle.dump(self.model, open(f"{path}/{model_name}.pkl", "wb"))
                logger.info(f"Model saved successfully at {path}.")
            else:
                logger.info("Model has not been trained yet.")

        except Exception as e:
            logger.error(f"Error in save: {e}")

    def preprocess_data_api(self, dataframe: pd.DataFrame, invoice_ids: list) -> pd.DataFrame | None:
        """Preprocess data from API request."""
        try:
            if "invoiceId" not in dataframe.columns:
                raise ValueError("Column 'invoiceId' is missing in the DataFrame.")

            # Remove invoice id to predict
            invoices_to_predict = dataframe[dataframe["invoiceId"].isin(invoice_ids)].drop(columns=["invoiceId"])

            if "Unnamed: 0" in dataframe.columns:
                invoices_to_predict = dataframe.drop(columns=["Unnamed: 0"])

            invoices_to_predict = self.scaler.fit_transform(invoices_to_predict)

            return invoices_to_predict

        except Exception as e:
            logger.error(f"Error in preprocess_data_api: {e}")
            return None
