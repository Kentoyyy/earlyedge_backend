from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import torch
import torch.nn as nn
import numpy as np
import joblib
import os

router = APIRouter()

class DyslexiaModel(nn.Module):
    def __init__(self):
        super(DyslexiaModel, self).__init__()
        input_dim = 26  # Updated to correct input features
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x

# Load model and tools
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = DyslexiaModel()
model_path = os.path.join(base_dir, "models", "dyslexia_letter_confusion_model.pth")
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Load encoders and scaler
le_question_type = joblib.load(os.path.join(base_dir, "models", "le_question_type.joblib"))
scaler = joblib.load(os.path.join(base_dir, "models", "scaler.joblib"))

class AnswerItem(BaseModel):
    question_type: str  # e.g., "matching_task" or "same_different_task"
    shown_letters: List[str]  # e.g., ["b", "d", "p", "q"]
    correct: int  # 1 or 0
    response_time_ms: float  # e.g., 1234.56

# One-hot encoding for shown_letters
def letters_to_multihot(shown_letters_list):
    all_letters = ['b', 'd', 'p', 'q', 'm', 'n', 'u', 't', 'f', 'c', 'o', 'h', 'k', 'v', 'w', 'x', 'z', 'y', 'a', 'e', 'i', 'l', 'j']
    return [1 if letter in shown_letters_list else 0 for letter in all_letters]

# Preprocessing
# Preprocessing
def preprocess_input(data: List[AnswerItem]) -> torch.Tensor:
    features = []
    for item in data:
        # Validate question_type
        if item.question_type not in le_question_type.classes_:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid question_type: {item.question_type}. Allowed values are: {list(le_question_type.classes_)}"
            )

        # Encode question_type
        q_type_enc = le_question_type.transform([item.question_type])[0]

        # One-hot encode shown_letters
        shown_letters_enc = letters_to_multihot(item.shown_letters)

        # Scale response_time_ms
        # Disable feature names check
        rt_scaled = scaler.transform([[item.response_time_ms]])[0][0]

        # Combine features
        features.append([item.correct, rt_scaled, q_type_enc] + shown_letters_enc)

    features_array = np.array(features, dtype=np.float32)
    return torch.tensor(features_array)
@router.post("/dyslexia/submit_answer/")
async def submit_answer(answers: List[AnswerItem]):
    try:
        inputs = preprocess_input(answers)
        with torch.no_grad():
            outputs = model(inputs)
            mean_confidence = outputs.mean().item()

        prediction = "dyslexic" if mean_confidence >= 0.5 else "non-dyslexic"
        next_question_id = len(answers) + 1 if len(answers) < 10 else None

        return {
            "prediction": prediction,
            "confidence": round(mean_confidence, 2),
            "next_question_id": next_question_id
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
