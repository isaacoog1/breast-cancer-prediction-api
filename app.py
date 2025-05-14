from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load artifacts
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

class Features(BaseModel):
    values: list[float]

@app.post("/predict")
def predict(data: Features):
    arr = np.array(data.values).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    pred = model.predict(arr_scaled)[0]
    proba = model.predict_proba(arr_scaled)[0].tolist()
    return {"prediction": int(pred), "probabilities": proba}