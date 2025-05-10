import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# 1. Load NEW dataset
df = pd.read_csv(r'C:\Users\omlan\OneDrive\Documents\GitHub\early_edge\backend\data\dyslexia_letter_dataset_10k.csv')

# 2. Encode target label
df['target'] = df['group'].apply(lambda x: 1 if x == 'dyslexic' else 0)

# 3. Feature engineering
# Encoding question type
le_question_type = LabelEncoder()
df['question_type_enc'] = le_question_type.fit_transform(df['question_type'])

# Multi-hot encoding for shown_letters
all_letters = ['b', 'd', 'p', 'q', 'm', 'n', 'u', 't', 'f', 'c', 'o', 'h', 'k', 'v', 'w', 'x', 'z', 'y', 'a', 'e', 'i', 'l', 'j']

def letters_to_multihot(shown_letters_list):
    return [1 if letter in shown_letters_list else 0 for letter in all_letters]

# Assuming 'shown_letters' column contains a list of letters
df['shown_letters_enc'] = df['shown_letters'].apply(lambda x: letters_to_multihot(x.split(',')))

# Features and target
features = df[['correct', 'response_time_ms', 'question_type_enc']]
features = features.join(pd.DataFrame(df['shown_letters_enc'].tolist(), columns=all_letters))  # Join multi-hot encoded letters
target = df['target']

# 4. Train/test split (with 10% validation split)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)

# 5. Scale numerical feature (response_time_ms)
scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled['response_time_ms'] = scaler.fit_transform(X_train[['response_time_ms']])
X_test_scaled['response_time_ms'] = scaler.transform(X_test[['response_time_ms']])

# 6. Torch Dataset
class LetterDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = LetterDataset(X_train_scaled, y_train)
test_dataset = LetterDataset(X_test_scaled, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class DyslexiaModel(nn.Module):
    def __init__(self):
        super(DyslexiaModel, self).__init__()
        input_dim = 1 + 1 + 1 + len(all_letters)  # 1 for correct, 1 for response_time_ms, 1 for question_type_enc, and 26 for multi-hot encoding
        self.fc1 = nn.Linear(input_dim, 64)  # Increased size of first layer for better capacity
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x

# Initialize model, loss, and optimizer
model = DyslexiaModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 8. Training Loop
for epoch in range(15):  # Train for 15 epochs
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = (outputs > 0.5).float()
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = running_correct / total * 100
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {acc:.2f}%")

# 9. Evaluation
model.eval()
test_correct = 0
test_total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        preds = (outputs > 0.5).float()
        test_correct += (preds == labels).sum().item()
        test_total += labels.size(0)

print(f"\nTest Accuracy: {test_correct/test_total*100:.2f}%")

# 10. Save model
os.makedirs(os.path.join(base_dir := os.path.dirname(__file__), '..', 'models'), exist_ok=True)
torch.save(model.state_dict(), os.path.join(base_dir, '..', 'models', 'dyslexia_letter_confusion_model.pth'))
print("✅ Saved model to ../models/dyslexia_letter_confusion_model.pth")

# 11. Save encoders and scaler
joblib.dump(le_question_type, os.path.join(base_dir, '..', 'models', 'le_question_type.joblib'))
joblib.dump(scaler, os.path.join(base_dir, '..', 'models', 'scaler.joblib'))
print("✅ Saved encoders and scaler to ../models/")
