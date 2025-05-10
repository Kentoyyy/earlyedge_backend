import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from torchmetrics.classification import Accuracy

# Define file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/dyslexia_reading_model.pth")
DATA_PATH = os.path.join(BASE_DIR, "../data/dyslexia_reading_dataset_10k.csv")

# Load the dataset
df = pd.read_csv(DATA_PATH)

# Encode categorical labels (Dyslexia Risk levels)
label_encoder = LabelEncoder()
df['dyslexia_risk'] = label_encoder.fit_transform(df['dyslexia_risk'])

# Standardize numerical features
numerical_columns = ['reading_time_seconds', 'correct_answers_count', 'retelling_mistakes_count', 'age']
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Features and target
X = df[numerical_columns]
y = df['dyslexia_risk']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# Define the Neural Network model
class DyslexiaModel(nn.Module):
    def __init__(self):
        super(DyslexiaModel, self).__init__()
        self.fc1 = nn.Linear(4, 128)  # Increased hidden units
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 3)  # 3 classes
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)  # No softmax here (because CrossEntropyLoss expects raw logits)
        return x

# Initialize model, loss, optimizer
model = DyslexiaModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 30  # Slightly increased epochs
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save the trained model
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# Evaluate the model
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    
    accuracy = accuracy_score(y_test_tensor, predicted)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    acc_metric = Accuracy(task='multiclass', num_classes=3)
    acc_metric.update(predicted, y_test_tensor)
    print(f"Accuracy (TorchMetrics): {acc_metric.compute():.2f}")
