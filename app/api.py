import os
import pandas as pd
import pickle
from enum import Enum

from fastapi import FastAPI
from pydantic import BaseModel, field_validator, Field

from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
#from langchain import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


class GoogleModels(Enum):
    GEMINI_2_0_FLASH = "gemini-2.0-flash"
    GEMINI_EMBEDDINGS = "models/text-embedding-004"

class ModelResponse(BaseModel):
    response: str

class InvoiceRisk(BaseModel):
    invoice_risk_predictions: List[int]

class UserInput(BaseModel):
    query: str = Field(..., description="User input")

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

def rag():
    google_api_key = os.getenv('GOOGLE_API_KEY')
    if google_api_key:
        print("Google API Key found!")
    else:
        print("Google API Key not found!")

    # Extract text from PDF
    relative_pdf_path = "data/attention_is_all_you_need.pdf"
    absolute_pdf_path = os.path.abspath(relative_pdf_path)
    pdf_loader = PyPDFLoader(absolute_pdf_path)
    pages = pdf_loader.load_and_split()

    # Create chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    context = "\n\n".join(str(p.page_content) for p in pages)
    texts = text_splitter.split_text(context)

    # Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model=GoogleModels.GEMINI_EMBEDDINGS.value,
        google_api_key=google_api_key
    )

    # Create vector index
    vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k": 5})

    # Model
    model = ChatGoogleGenerativeAI(
        model=GoogleModels.GEMINI_2_0_FLASH.value,
        api_key=google_api_key,
        temperature=0.2,
        convert_system_message_to_human=True
    )

    # Create qa chain
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, 
    just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. Always say 
    "thanks for asking!" at the end of the answer.
    {context}
    Question: {question}
    Helpful Answer:"""
    qa_chain_prompt = PromptTemplate.from_template(template)
    qa_chain = RetrievalQA.from_chain_type(
        model,
        retriever=vector_index,
        return_source_documents=True,
        chain_type_kwargs={"prompt": qa_chain_prompt}
    )
    return qa_chain


def initialize_model(app: FastAPI) -> None:
    @app.on_event("startup")
    async def startup():
        app.model = None
        app.model = pickle.load(open("model/xgb.pkl", "rb"))
        app.qa_chain = rag()


def init_app() -> FastAPI:
    app_ = FastAPI()
    initialize_model(app_)
    return app_
