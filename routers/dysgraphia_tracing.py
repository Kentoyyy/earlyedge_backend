import os
import torch
import torch.nn as nn
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


# --- 1. Define request/response schemas ---
class TraceRequest(BaseModel):
    letter: str
    drawing: str         # data URL (not used here, but could be used for future enhancements)
    duration: float
    accuracy: float      # Accuracy from the frontend tracing

class TraceResponse(BaseModel):
    label: str
    confidence: float
    duration_seconds: float  # Return the time for tracing in seconds
    accuracy: float         # Return the accuracy score for feedback

# --- 2. Define model architecture to match training ---
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(16, 8)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(8, 2)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

# --- 3. Load the trained model on startup ---
model_path = os.path.join(os.path.dirname(__file__), '../models/dysgraphia_tracing_model.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SimpleNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

LABELS = ['dysgraphia', 'non-dysgraphia']

# --- 4. Create router ---
router = APIRouter()

@router.post("/trace", response_model=TraceResponse)
async def trace_letter(req: TraceRequest):
    # Validate inputs
    if req.duration < 0 or not (0.0 <= req.accuracy <= 1.0):
        raise HTTPException(status_code=400, detail="Invalid duration or accuracy")

    # Prepare features for model: duration and accuracy
    features = torch.tensor([[req.duration, req.accuracy]], dtype=torch.float32).to(device)

    # Inference
    with torch.no_grad():
        logits = model(features)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(probs.argmax())
        label = LABELS[pred_idx]
        confidence = float(probs[pred_idx])

        # Threshold confidence level to ensure a more reliable prediction
        if confidence < 0.7:  # Adjust this threshold as needed
            label = "uncertain"  # or any other fallback

    # Return prediction label, confidence, duration, and accuracy for the frontend
    return TraceResponse(
        label=label,
        confidence=confidence,
        duration_seconds=req.duration,
        accuracy=req.accuracy
    )

