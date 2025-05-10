import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Paths
base_dir = os.path.dirname(__file__)
data_path = os.path.join(base_dir, '../data/number_understanding_dataset_10k.csv')
model_path = os.path.join(base_dir, '../models/dyscalculia_numberunderstanding.pth')

# Load dataset
data = pd.read_csv(data_path)

# Feature engineering
data['user_correct'] = (data['user_answer'] == data['correct_answer']).astype(int)

# Input features and label
X = data[['left_number', 'right_number', 'response_time_sec', 'user_correct']]
y = data['at_risk']

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler (if needed for backend inference consistency)
import joblib
scaler_path = os.path.join(base_dir, '../models/number_understanding_scaler.pkl')
joblib.dump(scaler, scaler_path)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# Model
class ImprovedModel(nn.Module):
    def __init__(self):
        super(ImprovedModel, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Initialize model, loss, optimizer
model = ImprovedModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
print("Training started...")
for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch + 1}/50], Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    _, predicted_classes = torch.max(predictions, 1)
    accuracy = accuracy_score(y_test_tensor, predicted_classes)
    print(f'✅ Accuracy on test set: {accuracy * 100:.2f}%')

# Save model
torch.save(model.state_dict(), model_path)
print(f'✅ Model saved to: {model_path}')
print(f'✅ Scaler saved to: {scaler_path}')
