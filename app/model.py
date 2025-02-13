import os
import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from xgboost import XGBClassifier

from typing import List


class Model:
    def __init__(self, sample_df: pd.DataFrame):
        """
        Initialize the class.
        """
        if not isinstance(sample_df, pd.DataFrame):
            raise ValueError("Pandas DataFrame is not provided. You must pass a valid DataFrame.")

        self.df = sample_df
        self.model = None
        # Initialize StandardScaler
        self.scaler = StandardScaler()

    def prepare_data(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """"
        Prepare data.

        Parameters:
        - df: pd.DataFrame, optional, input DataFrame.

        Returns:
        - pd.DataFrame, processed DataFrame.
        """
        if df is None:
            df = self.df

        # Create target column
        df["invoice_risk"] = np.where(df["overdueDays"] > 30, 1, 0)

        # Excluding columns for StandardScaler
        excluded_cols = ["invoiceId", "payerId", "invoice_risk"]
        feature_cols = df.columns.difference(excluded_cols)

        # Applying StandardScaler for other columns
        df[feature_cols] = self.scaler.fit_transform(df[feature_cols])

        return df

    def impute_missing(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Fill any missing values with the appropriate value.

        Parameters:
        - df: pd.DataFrame, optional, input DataFrame.

        Returns:
        - pd.DataFrame, DataFrame after imputation.
        """
        if df is None:
            df = self.df

        if df.isnull().any().any():
            print("DataFrame has missing values.")
            # For the purpose of the challenge, no code in this section.
        return df

    def fit(self, model=None) -> None:
        """
        Fit the model of your choice on the training data passed in the constructor, assuming it has
        been prepared by the functions prepare_data and impute_missing.

        Parameters:
        - model: object, optional, machine learning model. If None, XGBClassifier is used.
        """
        if model is None:
            model = XGBClassifier()

        # Prepare Data
        df = self.prepare_data()

        # Impute Missing Values
        df = self.impute_missing(df)

        # Splitting data in training and test
        X = df.drop(columns=["invoiceId", "invoice_risk"])
        y = df["invoice_risk"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=42)

        model.fit(X_train, y_train)

        self.model = model

        # Evaluate Model
        y_pred = model.predict(X_test)

        print("Classification Report:\n", classification_report(y_test, y_pred))

    def model_summary(self) -> str:
        """
        Create a short summary of the model you have fit.

        Returns:
        - str, model summary.
        """
        if self.model is not None:
            return f"Trained {type(self.model).__name__} on invoice risk prediction."
        else:
            return "Model has not been trained yet."

    def predict(self, invoice_ids: List[int]) -> pd.Series:
        """
        Make a set of predictions with your model. Assume the data has been prepared by the
        functions prepare_data and impute_missing.

        Parameters:
        - df: pd.DataFrame, optional, input DataFrame.

        Returns:
        - pd.Series, predicted invoice risk
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Use the fit method first.")

        invoices_to_predict = self.df[self.df["invoiceId"].isin(invoice_ids)].drop(
            columns=["invoiceId", "invoice_risk"])

        # Make predictions
        predictions = self.model.predict(invoices_to_predict)

        return pd.Series(predictions, name="predicted_invoice_risk")

    def save(self, path: str, model_name: str) -> None:
        """
        Save the model as .pkl

        Parameters:
        - path: str, file path to save the model.
        -model_name: str, Model name to be saved.
        """
        if self.model is not None:
            if not os.path.exists(path):
                os.makedirs(os.path.abspath(path))

            pickle.dump(self.model, open(f"{path}/{model_name}.pkl", "wb"))
            print(f"Model saved successfully at {path}.")
        else:
            print("Model has not been trained yet.")

    def preprocess_data_api(self, dataframe: pd.DataFrame, invoice_ids: list) -> pd.DataFrame:
        """
        Preprocess data from API request.

        Parameters:
        - dataframe: DataFrame, pandas dataframe to be processed.
        -invoice_ids: list, invoice ids to make a prediction.
        """
        invoices_to_predict = dataframe[dataframe["invoiceId"].isin(invoice_ids)].drop(columns=["Unnamed: 0", "invoiceId"])

        # Applying StandardScaler for other columns
        invoices_to_predict = self.scaler.fit_transform(invoices_to_predict)

        return invoices_to_predict
