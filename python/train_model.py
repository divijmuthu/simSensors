import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("imu_dataset.csv")
# grab data, sep features and labels
X_raw = df.drop("Label", axis=1).values
y_raw = df["Label"].values

# standardize each feature (Mean=0, Std=1) so the model learns faster
# crucial as 'Energy' (3.7) feature is much bigger than 'Variance' (0.001)
mean = X_raw.mean(axis=0)
std = X_raw.std(axis=0)
X_scaled = (X_raw - mean) / std

# convert data to pt tensors, float32 ok
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_raw, dtype=torch.long) # need long to classify
# split into Train (80%) and Test (20%)
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# create a neural network for our purposes
class ActivityClassifier(nn.Module):
    def __init__(self):
        super(ActivityClassifier, self).__init__()
        # input w/4 features, 16 hidden neurons
        self.layer1 = nn.Linear(4, 16) 
        self.relu = nn.ReLU()
        # Then hidden 16 neurons -> output between 2 classes (Sit, Walk)
        self.layer2 = nn.Linear(16, 2)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

model = ActivityClassifier()

# prep for training
criterion = nn.CrossEntropyLoss()  # good loss choice for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("--- Starting Training ---")
epochs = 100

for epoch in range(epochs):
    # forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    # backward pass, optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# now evaluate test performance 
print("\n--- Evaluation ---")
with torch.no_grad(): # Turn off gradient calc for speed
    test_outputs = model(X_test)
    # output score maps to prediction
    _, predicted = torch.max(test_outputs, 1)
    accuracy = accuracy_score(y_test, predicted)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

# save model, mean, std for live data scaling
torch.save({
    'model_state': model.state_dict(),
    'mean': mean,
    'std': std
}, "activity_model.pth")

print("Model saved to activity_model.pth")