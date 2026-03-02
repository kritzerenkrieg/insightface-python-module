from pydantic import BaseModel
from typing import List, Optional

class RegisterRequest(BaseModel):
    image: str
    name: str
    db: Optional[str] = "./db"

class RegisterResponse(BaseModel):
    message: str
    id: int

class RecognizeRequest(BaseModel):
    image: str
    db: Optional[str] = "./db"
    topk: Optional[int] = 1
    threshold: Optional[float] = 0.5

class RecognizeResponse(BaseModel):
    name: str
    id: int
    score: float

class RecognizeMultipleResponse(BaseModel):
    results: List[RecognizeResponse]