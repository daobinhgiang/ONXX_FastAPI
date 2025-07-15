from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
from PIL import Image
import io
from model import preprocess_image
import torch
import numpy as np

from CNN_model import FirstCNN
from schemas import PredictionResponse

import onnxruntime as ort

session = ort.InferenceSession("ONXX_model", providers=['CPUExecutionProvider'])
app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
possible_classes = ["cat", "dog"]  # Example


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Not an image.")
    image_bytes = await file.read()
    image_np = preprocess_image(image_bytes)

    inputs = {session.get_inputs()[0].name: image_np}
    outputs = session.run(None, inputs)
    probs = outputs[0]  # (1, num_classes)
    pred_idx = np.argmax(probs, axis=1)[0]
    pred_class = possible_classes[pred_idx]
    return {"prediction": pred_class, "possible_classes": possible_classes}
