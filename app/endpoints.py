import pandas as pd

from typing import Dict, Any
from app.api import InvoiceRisk, InvoicePayload

from app.model import Model
from app.api import init_app

from fastapi import status


app = init_app()


@app.post("/invoice_risk", response_model=InvoiceRisk, status_code=status.HTTP_200_OK)
def predict(payload: InvoicePayload) -> Dict[str, Any]:
    # Read data
    df = pd.read_csv("data/dataTest.csv")

    # Initialize class
    xgb_model = Model(df)

    # Data
    data = xgb_model.preprocess_data_api(df, payload.invoice_id)

    # Make predictions
    preds = app.model.predict(data)
    predictions = InvoiceRisk(invoice_risk_predictions=preds)
    predictions = predictions.model_dump()
    return predictions


@app.get("/health", status_code=status.HTTP_200_OK)
def health() -> Dict[str, str]:
    return {"healthcheck": "Everything OK!"}
