from pydantic import BaseModel
from typing import List

class PredictionRequest(BaseModel):
    # If you use image uploads, maybe have metadata here
    some_metadata: str

class PredictionResponse(BaseModel):
    prediction: str
    possible_classes: List[str]
