# runge_nn.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os, json, random

# ----- Reproducibility -----
SEED = 112652010
np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)

# ----- Prepare folder -----
os.makedirs("week_2", exist_ok=True)

# ----- Data: Runge function -----
def f(x):
    return 1.0 / (1.0 + 25.0 * x**2)

# train/val split
n_train, n_val = 200, 200
x_train = np.linspace(-1, 1, n_train).reshape(-1,1)
y_train = f(x_train)
x_val   = np.linspace(-1, 1, n_val).reshape(-1,1)
y_val   = f(x_val)

# to torch
xtr = torch.tensor(x_train, dtype=torch.float32)
ytr = torch.tensor(y_train, dtype=torch.float32)
xva = torch.tensor(x_val  , dtype=torch.float32)
yva = torch.tensor(y_val  , dtype=torch.float32)

# ----- Model -----
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x)

model = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)
epochs = 800

train_losses, val_losses = [], []

# ----- Train -----
for ep in range(epochs):
    model.train()
    pred = model(xtr)
    loss = criterion(pred, ytr)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # validation
    model.eval()
    with torch.no_grad():
        vpred = model(xva)
        vloss = criterion(vpred, yva)

    train_losses.append(float(loss))
    val_losses.append(float(vloss))

    if (ep+1) % 100 == 0:
        print(f"Epoch {ep+1:4d} | train {loss:.6f} | val {vloss:.6f}")

# ----- Metrics -----
with torch.no_grad():
    mse = float(criterion(vpred, yva))
    max_err = float(torch.max(torch.abs(vpred - yva)))
print(f"MSE={mse:.6e}, MaxError={max_err:.6e}")

# ----- Plots -----
# 1) Function vs Prediction
plt.figure()
plt.plot(x_val, y_val, label="True f(x)")
plt.plot(x_val, vpred.numpy(), label="NN prediction")
plt.xlabel("x"); plt.ylabel("y")
plt.title("Runge function approximation")
plt.legend()
plt.tight_layout()
plt.savefig("week_2/runge_fit.png", dpi=200)

# 2) Loss curves
plt.figure()
plt.plot(train_losses, label="Train loss")
plt.plot(val_losses, label="Validation loss")
plt.xlabel("epoch"); plt.ylabel("MSE")
plt.title("Loss curves")
plt.legend()
plt.tight_layout()
plt.savefig("week_2/loss_curves.png", dpi=200)

# 3) Save raw numbers for your report
with open("week_2/metrics.json", "w", encoding="utf-8") as fjson:
    json.dump({
        "seed": SEED,
        "epochs": epochs,
        "optimizer": "Adam",
        "lr": 1e-2,
        "architecture": "1-64-64-1 with Tanh",
        "train_final_mse": train_losses[-1],
        "val_final_mse": mse,
        "val_max_error": max_err
    }, fjson, indent=2)

np.savetxt("week_2/train_losses.txt", np.array(train_losses))
np.savetxt("week_2/val_losses.txt",   np.array(val_losses))

print("Saved: week_2/runge_fit.png, week_2/loss_curves.png, week_2/metrics.json")
