import os
import pandas as pd

from typing import Dict, Any
from app.api import InvoiceRisk, InvoicePayload, UserInput

from app.model import Model
from app.api import init_app

from fastapi import status

from langchain_google_genai import ChatGoogleGenerativeAI

# os.environ['GOOGLE_API_KEY'] =
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
print("GOOGLE_API_KEY: ", GOOGLE_API_KEY)
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

@app.post("/generate_response", status_code=status.HTTP_200_OK)
def generate_response(user_query: UserInput):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=GOOGLE_API_KEY)
    response = llm.invoke("What are the usecases of LLMs?")

    return {"response": response.content}



@app.get("/health", status_code=status.HTTP_200_OK)
def health() -> Dict[str, str]:
    return {"healthcheck": "Everything OK!"}
