from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import torch
import torch.nn as nn
import random
import string

router = APIRouter()

# === Load datasets ===
frontend_csv_path = './data/spellingfrontend_test.csv'
ground_truth_csv_path = './data/spelling_audio_dataset.csv'

try:
    frontend_df = pd.read_csv(frontend_csv_path)
    ground_truth_df = pd.read_csv(ground_truth_csv_path)
except FileNotFoundError as e:
    raise FileNotFoundError(f"CSV file not found: {e.filename}")


class AudioSpellingModel(nn.Module):
    def __init__(self):
        super(AudioSpellingModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=13, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()

        # Calculate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 13, 100)
            x = self.pool(self.relu(self.conv1(dummy_input)))
            self.flattened_size = x.view(1, -1).size(1)

        self.fc1 = nn.Linear(self.flattened_size, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.float()
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


# Load the trained model
model_path = './models/dyslexia_spelling_audio_model.pth'
model = AudioSpellingModel()
try:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
except FileNotFoundError:
    raise FileNotFoundError(f"Model file not found at {model_path}. Ensure the file exists.")

def word_to_tensor(word, max_len=100):  # Set max_len to 100
    alphabet = string.ascii_lowercase
    word = word.lower()
    indices = [alphabet.index(c) if c in alphabet else 0 for c in word[:max_len]]
    indices += [0] * (max_len - len(indices))  # Pad with zeros
    tensor = torch.tensor(indices, dtype=torch.float).unsqueeze(0).unsqueeze(0)  # [1, 1, len]
    tensor = tensor.repeat(1, 13, 1)  # Simulate 13 channels
    return tensor
# === Utility: Risk classification ===
def classify_risk(score):
    if score >= 0.7:
        return "High"
    elif score >= 0.4:
        return "Medium"
    else:
        return "Low"

@router.get("/get-audio")
async def get_audio():
    try:
        random_row = frontend_df.sample(1).iloc[0]
        audio_file = random_row['audio_file']
        correct_spelling = random_row['correct_word']
        print(f"Selected audio file: {audio_file}, Correct spelling: {correct_spelling}")
        return {"audio_file": audio_file, "correct_word": correct_spelling}
    except Exception as e:
        print(f"Error in /api/get-audio: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/validate-answer")
async def validate_answer(request: Request):
    try:
        # Ensure the request body is not empty
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="Request body is empty.")

        data = await request.json()
        print(f"Received data: {data}")

        user_answer = data.get('user_answer')
        audio_file = data.get('audio_file')

        if not user_answer or not audio_file:
            raise HTTPException(status_code=400, detail="Missing user_answer or audio_file in request.")

        # Normalize the audio_file path to match the dataset
        normalized_audio_file = f"audio/correct/{audio_file}" if not audio_file.startswith("audio/correct/") else audio_file

        # Look up the audio file in the ground_truth_df dataset
        correct_row = ground_truth_df[ground_truth_df['audio_file'] == normalized_audio_file]
        if correct_row.empty:
            return JSONResponse(status_code=404, content={"error": "Audio file not found in trained dataset."})

        correct_word = correct_row.iloc[0]['correct_spelling']
        is_correct = user_answer.strip().lower() == correct_word.strip().lower()

        # ML model part
        input_tensor = word_to_tensor(user_answer)
        with torch.no_grad():
            dyslexia_prob = model(input_tensor).item()
        risk = classify_risk(dyslexia_prob)

        return {
            "is_correct": is_correct,
            "user_answer": user_answer,
            "correct_word": correct_word,
            "dyslexia_score": round(dyslexia_prob, 2),
            "risk": risk
        }

    except Exception as e:
        print(f"Error in /api/validate-answer: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})