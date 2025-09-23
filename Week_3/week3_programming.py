# week3_programming.py
# Week 3: approximate Runge function AND its derivative with a tanh MLP

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os, json, random

# ----- Reproducibility -----
SEED = 112652010
np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)

# ----- Save folder -----
os.makedirs("week_3", exist_ok=True)

# ----- Runge function & derivative -----
def f(x):
    return 1.0 / (1.0 + 25.0 * x**2)

def fprime(x):
    return -50.0 * x / (1.0 + 25.0 * x**2)**2

# ----- Data (train/val) -----
n_train, n_val = 256, 128
x_train = np.linspace(-1, 1, n_train).reshape(-1, 1)
x_val   = np.linspace(-1, 1, n_val).reshape(-1, 1)

y_train = f(x_train)
y_val   = f(x_val)
yd_train = fprime(x_train)
yd_val   = fprime(x_val)

# to torch
xtr  = torch.tensor(x_train, dtype=torch.float32)
ytr  = torch.tensor(y_train, dtype=torch.float32)
ydtr = torch.tensor(yd_train, dtype=torch.float32)
xva  = torch.tensor(x_val, dtype=torch.float32)
yva  = torch.tensor(y_val, dtype=torch.float32)
ydva = torch.tensor(yd_val, dtype=torch.float32)

# ----- Model (same style as Week 2) -----
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
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 1200

# 想更重視導數就把 w_d 調大（例如 2 或 5）
w_f, w_d = 1.0, 1.0

# ----- Helper: forward + input-derivative -----
def forward_with_derivative(x: torch.Tensor, create_graph: bool):
    """
    回傳 (f_hat, fprime_hat)
    create_graph=True 用於訓練（需要反傳）；False 用於評估/繪圖（不需二階梯度）
    """
    x = x.clone().detach().requires_grad_(True)
    f_hat = model(x)              # (N,1)
    ones = torch.ones_like(f_hat)
    d_hat = torch.autograd.grad(
        outputs=f_hat, inputs=x, grad_outputs=ones,
        create_graph=create_graph, retain_graph=create_graph, only_inputs=True
    )[0]
    return f_hat, d_hat

# ----- Train loop -----
train_tot, val_tot, train_f, train_d, val_f, val_d = [], [], [], [], [], []

for ep in range(epochs):
    # train
    model.train()
    f_hat, d_hat = forward_with_derivative(xtr, create_graph=True)
    loss_f = criterion(f_hat, ytr)
    loss_d = criterion(d_hat, ydtr)
    loss = w_f*loss_f + w_d*loss_d

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # validate (注意：需要對輸入求導，不能用 no_grad 包住)
    model.eval()
    with torch.enable_grad():
        vf_hat, vd_hat = forward_with_derivative(xva, create_graph=False)
    vloss_f = criterion(vf_hat, yva)
    vloss_d = criterion(vd_hat, ydva)
    vloss   = w_f*vloss_f + w_d*vloss_d

    train_tot.append(loss.detach().item());   val_tot.append(vloss.detach().item())
    train_f.append(loss_f.detach().item());   val_f.append(vloss_f.detach().item())
    train_d.append(loss_d.detach().item());   val_d.append(vloss_d.detach().item())

    if (ep+1) % 100 == 0:
        print(f"Epoch {ep+1:4d} | train {loss:.6e} (f {loss_f:.2e}, d {loss_d:.2e}) | "
              f"val {vloss:.6e}")

# ----- Metrics on a dense grid -----
xx = np.linspace(-1, 1, 500).reshape(-1,1)
xx_t = torch.tensor(xx, dtype=torch.float32)

model.eval()
with torch.enable_grad():
    fpred, dpred = forward_with_derivative(xx_t, create_graph=False)

f_true  = f(xx).astype(np.float32)
df_true = fprime(xx).astype(np.float32)
f_pred  = fpred.detach().numpy().astype(np.float32)
df_pred = dpred.detach().numpy().astype(np.float32)

def mse_np(a,b): return float(np.mean((a-b)**2))
def maxerr_np(a,b): return float(np.max(np.abs(a-b)))

mse_f  = mse_np(f_pred, f_true)
mse_d  = mse_np(df_pred, df_true)
max_f  = maxerr_np(f_pred, f_true)
max_d  = maxerr_np(df_pred, df_true)

print(f"\nMSE(f)    = {mse_f:.6e}")
print(f"MSE(f')   = {mse_d:.6e}")
print(f"MaxErr(f) = {max_f:.6e}")
print(f"MaxErr(f')= {max_d:.6e}")

# ----- Plots -----
# 1) f vs prediction
plt.figure()
plt.plot(xx, f_true, label="True f(x)")
plt.plot(xx, f_pred, "--", label="NN f̂(x)")
plt.xlabel("x"); plt.ylabel("y"); plt.title("Function approximation")
plt.legend(); plt.tight_layout()
plt.savefig("week_3/runge_fit_f.png", dpi=200)

# 2) f' vs prediction
plt.figure()
plt.plot(xx, df_true, label="True f'(x)")
plt.plot(xx, df_pred, "--", label="NN f̂'(x)")
plt.xlabel("x"); plt.ylabel("y"); plt.title("Derivative approximation")
plt.legend(); plt.tight_layout()
plt.savefig("week_3/runge_fit_df.png", dpi=200)

# 3) Loss curves (total)
plt.figure()
plt.semilogy(train_tot, label="Train total")
plt.semilogy(val_tot,   label="Val total")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Total loss")
plt.legend(); plt.tight_layout()
plt.savefig("week_3/loss_total.png", dpi=200)

# 4) Loss curves (per component)
plt.figure()
plt.semilogy(train_f, label="Train f")
plt.semilogy(val_f,   label="Val f")
plt.xlabel("epoch"); plt.ylabel("MSE"); plt.title("Function loss")
plt.legend(); plt.tight_layout()
plt.savefig("week_3/loss_f.png", dpi=200)

plt.figure()
plt.semilogy(train_d, label="Train f'")
plt.semilogy(val_d,   label="Val f'")
plt.xlabel("epoch"); plt.ylabel("MSE"); plt.title("Derivative loss")
plt.legend(); plt.tight_layout()
plt.savefig("week_3/loss_df.png", dpi=200)

# 5) Save raw numbers for your report
with open("week_3/metrics.json", "w", encoding="utf-8") as fjson:
    json.dump({
        "seed": SEED,
        "epochs": epochs,
        "optimizer": "Adam",
        "lr": 1e-3,
        "architecture": "1-64-64-1 with Tanh",
        "weights": {"w_f": w_f, "w_d": w_d},
        "mse_f": mse_f, "mse_fprime": mse_d,
        "maxerr_f": max_f, "maxerr_fprime": max_d
    }, fjson, indent=2)

np.savetxt("week_3/train_total.txt", np.array(train_tot))
np.savetxt("week_3/val_total.txt",   np.array(val_tot))
np.savetxt("week_3/train_f.txt",     np.array(train_f))
np.savetxt("week_3/val_f.txt",       np.array(val_f))
np.savetxt("week_3/train_df.txt",    np.array(train_d))
np.savetxt("week_3/val_df.txt",      np.array(val_d))

print("Saved: week_3/runge_fit_f.png, runge_fit_df.png, loss_total.png, loss_f.png, loss_df.png, metrics.json")
