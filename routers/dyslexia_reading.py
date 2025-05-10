# routers/dyslexia_reading.py

import os
import torch
import torch.nn as nn
from fastapi import APIRouter
from pydantic import BaseModel
import numpy as np
from sklearn.preprocessing import LabelEncoder

router = APIRouter()

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "dyslexia_reading_model.pth")

# Updated Pydantic model to match frontend
class ReadingTestInput(BaseModel):
    transcript: str
    reading_time_seconds: float
    age: int
    expected_sentence: str
    language: str

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(32, 3)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        return x

# Load model and label encoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

label_encoder = LabelEncoder()
label_encoder.classes_ = ["High Risk", "Low Risk", "Moderate Risk"]

def calculate_correct_answers(expected: str, transcript: str) -> int:
    """Rough word match: how many words from expected sentence appear correctly in transcript."""
    expected_words = expected.lower().split()
    transcript_words = transcript.lower().split()

    correct = sum(1 for word in expected_words if word in transcript_words)
    return correct

def calculate_retelling_mistakes(expected: str, transcript: str) -> int:
    """Estimate mistakes: number of missing words."""
    expected_words = set(expected.lower().split())
    transcript_words = set(transcript.lower().split())
    missing_words = expected_words - transcript_words
    return len(missing_words)

@router.post("/reading", summary="Predict Dyslexia Risk from Reading Test")
async def predict_reading_risk(input_data: ReadingTestInput):
    reading_time_seconds = input_data.reading_time_seconds
    age = input_data.age
    transcript = input_data.transcript
    expected_sentence = input_data.expected_sentence

    # Calculate derived features
    correct_answers_count = calculate_correct_answers(expected_sentence, transcript)
    retelling_mistakes_count = calculate_retelling_mistakes(expected_sentence, transcript)

    features = [
        reading_time_seconds,
        correct_answers_count,
        retelling_mistakes_count,
        age
    ]
    features_tensor = torch.tensor([features], dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(features_tensor)
    _, predicted_class = torch.max(outputs, 1)
    predicted_class = int(predicted_class.cpu().item())  # force plain Python int
    risk_label = label_encoder.inverse_transform([predicted_class])[0]

    return {"risk_prediction": risk_label}
