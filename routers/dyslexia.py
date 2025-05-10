from fastapi import APIRouter, HTTPException
import torch
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

router = APIRouter()

# Define Base Directory (Relative Paths)
BASE_DIR = Path(__file__).resolve().parent.parent

# Model & Vectorizer Paths
model_path = BASE_DIR / "models/dyslexia_model.pth"
vectorizer_path = BASE_DIR / "training_models/vectorizer.pkl"

# Load Vectorizer
try:
    vectorizer = joblib.load(vectorizer_path)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Failed to load vectorizer: {str(e)}")

# Define Model Class
class DyslexiaModel(torch.nn.Module):
    def __init__(self, input_size):
        super(DyslexiaModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 64)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(64, 2)  # Two classes: No Dyslexia (0), Dyslexia (1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Load Model
input_size = len(vectorizer.get_feature_names_out())
model = DyslexiaModel(input_size)
try:
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()  # Set model to evaluation mode
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

# Get Dyslexia Test Questions
@router.get("/questions")
def get_questions():
    dataset_path = BASE_DIR / "data/dyslexia_dataset.csv"
    if not dataset_path.exists():
        raise HTTPException(status_code=500, detail="Dataset file is missing.")

    df = pd.read_csv(dataset_path)
    if "Question" not in df.columns:
        raise HTTPException(status_code=500, detail="Dataset is missing 'Question' column.")

    return {"questions": df["Question"].tolist()}

# Analyze User Response
@router.post("/analyze")
def analyze_response(data: dict):
    question = data.get("question", "").strip()
    response = data.get("response", "").strip()

    if not question or not response:
        raise HTTPException(status_code=400, detail="Both question and response are required.")

    try:
        input_text = [question + " " + response]
        input_vector = vectorizer.transform(input_text).toarray()
        input_tensor = torch.tensor(input_vector, dtype=torch.float32).reshape(1, -1)

        with torch.no_grad():
            prediction = model(input_tensor)
            predicted_label = torch.argmax(prediction, dim=1).item()

        return {"result": "High Risk of Dyslexia" if predicted_label == 1 else "No Dyslexia Risk"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing response: {str(e)}")
