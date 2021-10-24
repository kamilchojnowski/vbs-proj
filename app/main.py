import boto3
import io
import os
import pandas
import pathlib

from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Depends
from tensorflow.keras.models import load_model

from app.config import Settings
from .model.model import lstm_model, get_data_to_predict, get_labels


BASE_PATH = pathlib.Path(__file__).resolve().parent
AI_MODEL = None

app = FastAPI()

@lru_cache()
def get_settings():
    return Settings()


@app.get("/")
def health():
    return {"Hello": "World"}


@app.on_event("startup")
def on_startup():
    global AI_MODEL
    settings = get_settings()
    MODEL_PATH = BASE_PATH / settings.MODEL_DIR / settings.MODEL_FILE
    if MODEL_PATH.exists():
        model = lstm_model(settings.MODEL_SHAPE)
        model.load_weights(MODEL_PATH)
        AI_MODEL = model


@app.post("/predict/")
def predict(file: UploadFile = File(...)):
    settings = get_settings()
    pq_file = io.BytesIO(file.file.read())
    data = pandas.read_parquet(pq_file)
    pred_data = get_data_to_predict(data)
    prediction = AI_MODEL.predict(pred_data)
    return get_labels(prediction, data, settings)

