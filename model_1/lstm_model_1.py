import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import time

x = np.linspace(0, 100, 1000)
y = np.sin(x)

rng = np.random.default_rng (42) # fixed seed
noise_std = 0.10 			# try 0.05 to 0.20
y = y + rng.normal(0.0, noise_std, size=y . shape)

# Prepare sequences: 50 timesteps to predict the next value
seq_length = 100

X, Y = [], []
for i in range(len(y) - seq_length):
    X.append(y[i:i+seq_length])
    Y.append(y[i+seq_length])

X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)  # (samples, seq_len, 1)
Y = torch.tensor(np.array(Y), dtype=torch.float32).unsqueeze(-1)  # (samples, 1)

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = LSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 200
start_time = time.perf_counter()

for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, Y)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 50 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.8f}')


model.eval()
end_time = time.perf_counter()
elapsed_time = (end_time - start_time)
print(f"Elapsed time: {elapsed_time:.2f} seconds")

predictions = []
input_seq = X[-1:].clone()
with torch.no_grad():
    preds = model(X)
    for _ in range(200):
        # Predict the next value
        next_val = model(input_seq)
        # Store it
        predictions.append(next_val.item())
        # Slide the window forward: drop first, append new
        input_seq = torch.cat(
        [input_seq[:, 1:, :], next_val.unsqueeze(0)], dim=1)

    # Compute R^2 (coefficient of determination)
    ss_res = torch.sum((Y - preds) ** 2).item()
    ss_tot = torch.sum((Y - torch.mean(Y)) ** 2).item()
    r2 = 1 - ss_res / ss_tot
    print(f"\nAccuracy Score: {r2 * 100:.2f}%")

with torch.no_grad():
    preds = model(X).squeeze(-1).numpy()
    targets = Y.squeeze(-1).numpy()

# Plot first 200 points for readability
n_plot = 200

plt.figure()
plt.plot(targets[:n_plot], label="True")
plt.plot(preds[:n_plot], label="Predicted")
plt.xlabel("Time step")
plt.ylabel("Value")
plt.title("LSTM prediction vs ground truth")
plt.legend()
plt.tight_layout()
plt.savefig("pred_vs_true.png")
print("saved pred_vs_true.png")