import os
import pandas as pd
from google.cloud import secretmanager

from app.model import Model

import uvicorn

from app.endpoints import app

def get_secret(secret_name: str) -> str:
    """Retrieve a secret value from Google Secret Manager."""
    project_id = os.getenv("PROJECT_ID", None)  # Replace with your GCP Project ID
    client = secretmanager.SecretManagerServiceClient()
    secret_path = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
    response = client.access_secret_version(name=secret_path)
    return response.payload.data.decode("UTF-8")


def main():
    # Retrieve the Gemini API Key from Secret Manager
    GEMINI_API_KEY = get_secret("gemini-sa-key")
    print("GEMINI_API_KEY:")
    print(GEMINI_API_KEY)

    dataframe = pd.read_csv("data/dataTest.csv")
    dataframe = dataframe.drop(columns=["Unnamed: 0"])

    # Initialize Class
    run_model = Model(dataframe)

    # Fit model
    run_model.fit()

    # Save Model
    path = "model"
    model_name = "xgb"

    run_model.save(path=path, model_name=model_name)


if __name__ == "__main__":
    main()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
    )
