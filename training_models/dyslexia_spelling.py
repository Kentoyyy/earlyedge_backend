import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import pandas as pd
import librosa
import numpy as np
import os
from sklearn.metrics import accuracy_score

# ======= Setup paths =======
base_dir = os.path.abspath(os.path.dirname(__file__))
data_path = os.path.join(base_dir, '../data/spelling_audio_dataset.csv')
model_path = os.path.join(base_dir, '../models/dyslexia_spelling_audio_model.pth')

# ======= Dataset =======
class AudioDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        audio_path = os.path.abspath(os.path.join(base_dir, '..', row['audio_file']))

        # Assign label based on folder in path: correct = 1, incorrect = 0
        label = 1 if 'correct' in row['audio_file'].lower() else 0

        # Load audio and extract MFCCs
        y, sr = librosa.load(audio_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        # Pad or truncate to fixed length
        max_len = 100
        if mfcc.shape[1] < max_len:
            pad_width = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_len]

        return torch.tensor(mfcc, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# ======= Model =======
class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.conv1 = nn.Conv1d(13, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()

        # Dummy input to calculate flattened size
        dummy_input = torch.zeros(1, 13, 100)  # (batch, channels, sequence)
        x = self.pool(self.relu(self.conv1(dummy_input)))
        self.flattened_size = x.view(1, -1).size(1)

        self.fc1 = nn.Linear(self.flattened_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x)).squeeze()


# ======= Training =======
def train():
    df = pd.read_csv(data_path)
    full_dataset = AudioDataset(df)

    # Split dataset into train/test
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8)

    model = AudioClassifier()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(torch.float32)
            labels = labels.to(torch.float32)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {running_loss:.4f}")

    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")

    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(torch.float32)
            predictions = model(inputs).squeeze()
            predicted_classes = (predictions > 0.5).int()
            all_preds.extend(predicted_classes.tolist())
            all_labels.extend(labels.int().tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f'✅ Accuracy on test set: {accuracy * 100:.2f}%')

if __name__ == "__main__":
    train()
