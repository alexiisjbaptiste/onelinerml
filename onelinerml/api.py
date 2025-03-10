from fastapi import FastAPI, File, UploadFile, HTTPException
from .train import train
import pandas as pd
from io import StringIO

app = FastAPI()

@app.post("/train")
async def train_endpoint(file: UploadFile = File(...), model: str = "linear_regression", target_column: str = "target"):
    try:
        contents = await file.read()
        data = pd.read_csv(StringIO(contents.decode("utf-8")))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid file format or content.")
    
    model_instance, metrics = train(data, model=model, target_column=target_column)
    return {"metrics": metrics}

@app.post("/predict")
async def predict_endpoint(model_name: str, data: dict):
    raise HTTPException(status_code=501, detail="Prediction endpoint not implemented yet.")
