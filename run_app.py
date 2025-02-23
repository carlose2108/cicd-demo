import os
import pandas as pd

from app.model import Model

import uvicorn

from app.endpoints import app

def main():
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
