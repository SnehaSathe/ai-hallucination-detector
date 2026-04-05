from pydantic import BaseModel

class DetectionRequest(BaseModel):
    context: str
    answer: str

class DetectionResponse(BaseModel):
    score: float
    label: str