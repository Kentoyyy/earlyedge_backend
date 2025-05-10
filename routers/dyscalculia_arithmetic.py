import torch
import torch.nn as nn
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

# ======= Load Model =======
class ArithmeticClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(5, 32),  # Model still expects only 5 inputs
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

model = ArithmeticClassifier()
model.load_state_dict(torch.load('./models/dyscalculia_arithmetic.pth'))
model.eval()

router = APIRouter(prefix="/api/arithmetic")

class Attempt(BaseModel):
    op1: int
    op2: int
    operation: str
    user_choice: int  # 0 for correct, 1 for incorrect
    response_time: float

class SummaryRequest(BaseModel):
    attempts: List[Attempt]

@router.post("/summary")
async def calculate_summary(request: SummaryRequest):
    try:
        total_correct = 0
        total_time = 0
        slow_count = 0
        fast_count = 0
        moderate_count = 0
        risk_count = 0

        for attempt in request.attempts:
            # Map operation to integer value
            operation_dict = {'+': 0, '-': 1, '*': 2, '/': 3}
            features = [
                attempt.op1,
                attempt.op2,
                operation_dict.get(attempt.operation, -1),
                attempt.user_choice,
                attempt.response_time
            ]

            features_tensor = torch.tensor([features], dtype=torch.float32)

            with torch.no_grad():
                # Model predicts risk using the features provided.
                prediction = model(features_tensor).squeeze()
                is_at_risk = 1 if prediction > 0.5 else 0

            # You might choose to ignore the risk prediction if the answer is correct.
            # For example, if the user_choice is 0 (correct), you could force is_at_risk to 0:
            if attempt.user_choice == 0:
                is_at_risk = 0

            # Determine response speed counts
            if attempt.response_time > 3:
                slow_count += 1
            elif attempt.response_time < 1.5:
                fast_count += 1
            else:
                moderate_count += 1

            risk_count += is_at_risk
            if attempt.user_choice == 0:  # Correct answer
                total_correct += 1

            total_time += attempt.response_time

        total_attempts = len(request.attempts)
        avg_time = total_time / total_attempts

        # Determine overall risk
        if total_correct == total_attempts:
            overall_risk = "No Risk"
        else:
            risk_ratio = risk_count / total_attempts
            if risk_ratio < 0.33:
                overall_risk = "Mild Risk"
            elif risk_ratio < 0.66:
                overall_risk = "Moderate Risk"
            else:
                overall_risk = "At Risk"

        # Determine speed category based on counts
        speed_category = "Slow" if slow_count > fast_count and slow_count > moderate_count else \
                         "Fast" if fast_count > slow_count and fast_count > moderate_count else \
                         "Moderate"

        return {
            "total_correct": total_correct,
            "average_time": avg_time,
            "overall_risk": overall_risk,
            "speed_category": speed_category,
            "risk_count": risk_count,
            "total_attempts": total_attempts
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in summary calculation: {e}")
