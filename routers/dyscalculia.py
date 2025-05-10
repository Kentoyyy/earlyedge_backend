from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import pandas as pd
import os
import random
import torch.nn.functional as F

router = APIRouter()

# Shared Model Class
class ImprovedModel(nn.Module):
    def __init__(self, input_size=4, output_size=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# === Number Understanding Model ===
number_model_path = "models/dyscalculia_numberunderstanding.pth"
number_model = ImprovedModel(input_size=4, output_size=2)

try:
    number_model.load_state_dict(torch.load(number_model_path, map_location=torch.device("cpu")))
    number_model.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load number understanding model: {str(e)}")

dataset_path = os.path.join(os.path.dirname(__file__), "../data/number_understanding_dataset_10k.csv")
try:
    dataset = pd.read_csv(dataset_path)
except Exception as e:
    raise RuntimeError(f"Failed to load dataset: {str(e)}")


class PredictionInput(BaseModel):
    left_number: float
    right_number: float
    response_time_sec: float
    user_correct: int

def normalize_input(left, right, time, correct):
    return [left / 100, right / 100, time / 10, correct]


@router.get("/getQuestions")
async def get_questions():
    try:
        row = dataset.sample(n=1).iloc[0]
        return {
            "question_type": row["question_type"],
            "left_number": int(row["left_number"]),
            "right_number": int(row["right_number"]),
            "correct_answer": row["correct_answer"],
            "at_risk": int(row["at_risk"])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching question: {str(e)}")


@router.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        inputs = normalize_input(
            input_data.left_number,
            input_data.right_number,
            input_data.response_time_sec,
            input_data.user_correct,
        )
        tensor_input = torch.tensor([inputs], dtype=torch.float32)

        with torch.no_grad():
            output = number_model(tensor_input)
            probabilities = torch.softmax(output, dim=1)
            confidence = round(probabilities[0][1].item(), 4)
            _, predicted = torch.max(output, 1)

        is_at_risk = int(predicted.item())

        rt = input_data.response_time_sec
        if rt < 3:
            speed = "Fast"
            message = "The child responded quickly. This may indicate good number recognition."
        elif rt <= 6:
            speed = "Moderate"
            message = "The response time is within a normal range."
        else:
            speed = "Slow"
            message = "The child took longer to respond. This might indicate difficulty in understanding numbers."

        return {
            "at_risk": is_at_risk,
            "result": "At Risk for Learning Difficulty" if is_at_risk else "Not At Risk",
            "confidence": confidence,
            "response_time_sec": rt,
            "speed_category": speed,
            "speed_message": message
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


