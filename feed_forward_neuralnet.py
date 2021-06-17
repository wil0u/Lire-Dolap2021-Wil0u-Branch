from itertools import combinations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out

train_data = pd.read_csv("trainingSetVectors.csv")
train_data = np.asarray(train_data)
y_train = np.asarray([train_data[:, -1]])
train_data = train_data[:,1:-1]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(24, 50, 1).to(device)

# Loss and optimizer
learning_rate = 0.001
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
train_data = torch.from_numpy(train_data).to(dtype=torch.float).to(device)
y_train = torch.from_numpy(y_train).to(dtype=torch.float).to(device)
y_train = torch.transpose(y_train, -1, 0)
num_epochs = 1000
# Train the model
for epoch in range(num_epochs):

    # Forward pass
    outputs = model(train_data)
    # if y would be one-hot, we must apply
    # labels = torch.max(labels, 1)[1]
    loss = criterion(outputs, y_train)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        print(outputs[0:15])