from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import joblib
import pandas as pd
import numpy as np

# 1️⃣ Initialize FastAPI App
app = FastAPI()

# 2️⃣ Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust for your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3️⃣ Load the trained model & vectorizer
model_path = r"C:\Users\omlan\OneDrive\Documents\GitHub\early_edge\backend\models\dyslexia_model.pth"
vectorizer_path = "vectorizer.pkl"

class DyslexiaModel(torch.nn.Module):
    def __init__(self, input_size):
        super(DyslexiaModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 64)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(64, 2)  # Two classes (0 = No Dyslexia, 1 = Dyslexia)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Load model & vectorizer
vectorizer = joblib.load(vectorizer_path)
dummy_input_size = len(vectorizer.get_feature_names_out())  # Get input size
model = DyslexiaModel(dummy_input_size)
model.load_state_dict(torch.load(model_path))
model.eval()  # Set to evaluation mode

# 4️⃣ API Route: Get Dyslexia Test Questions
@app.get("/api/dyslexia/questions")
def get_questions():
    csv_path = r"C:\Users\omlan\OneDrive\Documents\GitHub\early_edge\backend\data\dyslexia_dataset.csv"
    df = pd.read_csv(csv_path)
    if "Question" not in df.columns:
        raise HTTPException(status_code=500, detail="Dataset is missing 'Question' column.")
    
    return {"questions": df["Question"].tolist()}

# 5️⃣ API Route: Analyze User Response
@app.post("/api/dyslexia/analyze")
def analyze_response(data: dict):
    question = data.get("question", "")
    response = data.get("response", "")

    if not question or not response:
        raise HTTPException(status_code=400, detail="Both question and response are required.")

    # Process input with TF-IDF
    input_text = [question + " " + response]
    input_vector = vectorizer.transform(input_text).toarray()
    input_tensor = torch.tensor(input_vector, dtype=torch.float32)

    # Get model prediction
    with torch.no_grad():
        prediction = model(input_tensor)
        predicted_label = torch.argmax(prediction, dim=1).item()

    result_text = "High Risk of Dyslexia" if predicted_label == 1 else "No Dyslexia Risk"
    return {"result": result_text}

# 6️⃣ Run FastAPI Server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
