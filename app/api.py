import pandas as pd
import pickle

from fastapi import FastAPI
from pydantic import BaseModel, field_validator, Field

from typing import List


class InvoiceRisk(BaseModel):
    invoice_risk_predictions: List[int]


class InvoicePayload(BaseModel):
    invoice_id: List[int] = Field(..., description="")
    country: str = Field(..., description="")

    @field_validator("invoice_id")
    def validate_invoice_id(cls, value):

        invoice_ids = pd.read_csv("data/dataTest.csv")
        valid_invoice_ids = invoice_ids["invoiceId"].tolist()

        if not all(item in valid_invoice_ids for item in value):
            raise ValueError(f"Invalid invoice id. Must be one of: {valid_invoice_ids}")
        return value

    @field_validator("country")
    def validate_country(cls, value):
        if value not in ["CL", "MX"]:
            raise ValueError("Invalid country. Must be 'CL' or 'MX'.")
        return value


def initialize_model(app: FastAPI) -> None:
    @app.on_event("startup")
    async def startup():
        app.model = None
        app.model = pickle.load(open("model/xgb.pkl", "rb"))


def init_app() -> FastAPI:
    app_ = FastAPI()
    initialize_model(app_)
    return app_
