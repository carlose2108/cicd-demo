import os
import pandas as pd

from typing import Dict, Any
from app.api import InvoiceRisk, InvoicePayload, UserInput, ModelResponse

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

@app.post("/generate_response", response_model=ModelResponse, status_code=status.HTTP_200_OK)
def generate_response(user_query: UserInput):
    """
    Generate response from RAG system
    :param user_query:
    :return:
    """
    response = app.qa_chain.invoke({"query": user_query.query})
    response = ModelResponse(response=response['result'])
    return response


@app.get("/health", status_code=status.HTTP_200_OK)
def health() -> Dict[str, str]:
    return {"healthcheck": "Everything OK!"}
