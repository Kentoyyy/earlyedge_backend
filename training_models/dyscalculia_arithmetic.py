import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader

# ======= Setup paths =======
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, '../data/arithmetic_data_1k.csv')
model_path = os.path.join(base_dir, '../models/dyscalculia_arithmetic.pth')

# ======= Load Dataset =======
df = pd.read_csv(data_path)

# ======= Feature Engineering =======
df['op1'] = df['question'].str.extract(r'(\d+)').astype(int)
df['op2'] = df['question'].str.extract(r'[\+\-\*/] (\d+)').astype(int)
df['operation'] = df['question'].str.extract(r'(\+|\-|\*|\/)')

# Encode operations
op_encoder = LabelEncoder()
df['operation'] = op_encoder.fit_transform(df['operation'])

# Encode choices
df['user_choice'] = df['user_choice'].apply(lambda x: 0 if x == 'choice_1' else 1)
df['correct_choice'] = df['correct_choice'].apply(lambda x: 0 if x == 'choice_1' else 1)

# ======= Add response_time as a feature =======
# Adding the response_time column as part of the features
df['response_time'] = df['response_time'].astype(float)  # Ensure it's in float for processing

# ======= Prepare Data =======
# Include response_time in the features (X)
X = df[['op1', 'op2', 'operation', 'user_choice', 'response_time']]
y = df['is_correct'].astype(int)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)

# ======= Dataset Class =======
class ArithmeticDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_data = ArithmeticDataset(X_train, y_train)
test_data = ArithmeticDataset(X_test, y_test)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

# ======= Model =======
class ArithmeticClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Adjusting input layer to match 5 features (including response_time)
        self.model = nn.Sequential(
            nn.Linear(5, 32),  # 5 features: op1, op2, operation, user_choice, response_time
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

model = ArithmeticClassifier()

# ======= Training Setup =======
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ======= Training Loop =======
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0
    for features, labels in train_loader:
        outputs = model(features).squeeze()
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

# ======= Save Model =======
os.makedirs(os.path.dirname(model_path), exist_ok=True)
torch.save(model.state_dict(), model_path)
print(f"✅ Model saved at: {model_path}")

# ======= Evaluation =======
model.eval()
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.int)

with torch.no_grad():
    predictions = model(X_test_tensor).squeeze()
    predicted_classes = (predictions > 0.5).int()
    accuracy = accuracy_score(y_test_tensor, predicted_classes)
    print(f'✅ Accuracy on test set: {accuracy * 100:.2f}%')
