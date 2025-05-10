import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
import os

# 1️⃣ Load the dataset
csv_path = r"C:\Users\omlan\OneDrive\Documents\GitHub\early_edge\backend\data\dyslexia_dataset.csv"
df = pd.read_csv(csv_path)

# 2️⃣ Check if 'Label' column exists
if "Label" not in df.columns:
    print("⚠️ WARNING: 'Label' column not found! Generating random labels...")
    df["Label"] = np.random.randint(0, 2, size=len(df))  # Random labels (0 or 1)

# 3️⃣ Preprocessing - TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_text = vectorizer.fit_transform(df["Question"] + " " + df["Correct_Response"] + " " + df["Incorrect_Response"]).toarray()

# Save vectorizer for future inference
joblib.dump(vectorizer, "vectorizer.pkl")

# Labels (0 = No Dyslexia, 1 = Possible Dyslexia)
y_labels = df["Label"].values  

# 4️⃣ Split dataset into Train & Test sets
X_train, X_test, y_train, y_test = train_test_split(X_text, y_labels, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 5️⃣ Define the PyTorch Model
class DyslexiaModel(nn.Module):
    def __init__(self, input_size):
        super(DyslexiaModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)  # Output: 2 classes (0 = No Dyslexia, 1 = Dyslexia)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)  # No Softmax here; handled by CrossEntropyLoss
        return x

# 6️⃣ Initialize Model
model = DyslexiaModel(input_size=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7️⃣ Training Loop
epochs = 10
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# 8️⃣ Save the trained model
model_path = r"C:\Users\omlan\OneDrive\Documents\GitHub\early_edge\backend\models\dyslexia_model.pth"
os.makedirs(os.path.dirname(model_path), exist_ok=True)
torch.save(model.state_dict(), model_path)

print("✅ Model trained and saved successfully!")
