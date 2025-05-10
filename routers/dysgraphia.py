from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import List
from torchvision import transforms
from PIL import Image
import torch
import io
import torch.nn.functional as F

from training_models.train_dysgraphia import DysgraphiaClassifier

router = APIRouter()

# Load trained model
model_path = "C:/Users/omlan/OneDrive/Documents/GitHub/early_edge/backend/models/dysgraphia_model.pth"
model = DysgraphiaClassifier()
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

@router.post("/predict")
async def predict(files: List[UploadFile] = File(...)): 
    if len(files) != 3:
        raise HTTPException(status_code=400, detail="Please upload exactly 3 images.")

    predictions = []

    labels = ["Dysgraphic", "Non-Dysgraphic"]

    for file in files:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = transform(image).unsqueeze(0)  # Add batch dimension

        output = model(image)
        softmax_output = F.softmax(output, dim=1)
        _, predicted_class = torch.max(output, 1)

        predicted_index = predicted_class.item()
        confidence = float(softmax_output[0][predicted_index].detach().cpu().numpy())
        prediction_label = labels[predicted_index] if predicted_index < len(labels) else "Unknown Classification"

        # Severity Mapping
        if 0.01 <= confidence <= 0.25:
            severity_level = "Mild"
        elif 0.26 <= confidence <= 0.50:
            severity_level = "Moderate"
        elif 0.51 <= confidence <= 0.75:
            severity_level = "Severe"
        elif 0.76 <= confidence <= 1.0:
            severity_level = "Profound"
        else:
            severity_level = "No significant impairment detected"

        predictions.append({
            "Filename": file.filename,
            "Prediction": prediction_label,
            "Confidence": confidence,
            "Severity": severity_level
        })

    return {"Results": predictions}