import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path

import os
print("Current working directory:", os.getcwd())


# ================================
# 1. Load datasets
# ================================
cls_path = Path("classification.csv")
reg_path = Path("regression.csv")

df_cls = pd.read_csv(cls_path)
df_reg = pd.read_csv(reg_path)

# infer columns
def infer_columns_for_classification(df):
    label_candidates = ["label", "y", "target", "class", "cls"]
    lbl = None
    for c in df.columns:
        if c.lower() in label_candidates:
            lbl = c
            break
    if lbl is None:
        last = df.columns[-1]
        if df[last].dropna().astype(float).isin([0, 1]).all():
            lbl = last
    num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    if lbl and lbl in num_cols:
        X_cols = [c for c in num_cols if c != lbl]
    else:
        if lbl is None:
            lbl = df.columns[-1]
        X_cols = [c for c in num_cols if c != lbl]
    return X_cols, lbl

def infer_columns_for_regression(df):
    y_candidates = ["y", "label", "target", "value", "temperature", "temp", "z"]
    y_col = None
    for c in df.columns:
        if c.lower() in y_candidates:
            y_col = c
            break
    if y_col is None:
        y_col = df.columns[-1]
    num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    X_cols = [c for c in num_cols if c != y_col]
    return X_cols, y_col

Xc_cols, yc_col = infer_columns_for_classification(df_cls)
Xr_cols, yr_col = infer_columns_for_regression(df_reg)

# ================================
# 2. Helper functions
# ================================
def train_test_split(X, y, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    m = X.shape[0]
    idx = np.arange(m)
    rng.shuffle(idx)
    cut = int(m * (1 - test_ratio))
    tr, te = idx[:cut], idx[cut:]
    return X[tr], X[te], y[tr], y[te]

# ================================
# 3. GDA Implementation
# ================================
@dataclass
class GDA:
    mu0: np.ndarray = None
    mu1: np.ndarray = None
    sigma: np.ndarray = None
    phi: float = None
    theta: np.ndarray = None
    theta0: float = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        m, n = X.shape
        self.phi = y.mean()
        X0, X1 = X[y == 0], X[y == 1]
        self.mu0, self.mu1 = X0.mean(axis=0), X1.mean(axis=0)
        sigma = np.zeros((n, n))
        for xi, yi in zip(X, y):
            mui = self.mu1 if yi == 1 else self.mu0
            diff = (xi - mui).reshape(-1, 1)
            sigma += diff @ diff.T
        self.sigma = sigma / m
        sigma_inv = np.linalg.pinv(self.sigma)
        self.theta = sigma_inv @ (self.mu1 - self.mu0)
        self.theta0 = (
            -0.5 * self.mu1.T @ sigma_inv @ self.mu1
            + 0.5 * self.mu0.T @ sigma_inv @ self.mu0
            + np.log(self.phi / (1 - self.phi + 1e-12) + 1e-12)
        )
        return self

    def predict_proba(self, X: np.ndarray):
        logits = X @ self.theta + self.theta0
        return 1 / (1 + np.exp(-logits))

    def predict(self, X: np.ndarray, threshold: float = 0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

# ================================
# 4. Train GDA
# ================================
Xc, yc = df_cls[Xc_cols].values, df_cls[yc_col].values.astype(int)
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_ratio=0.2, seed=1126)

gda = GDA().fit(Xc_tr, yc_tr)
yc_pred = gda.predict(Xc_te)
acc = (yc_pred == yc_te).mean()
print(f"GDA Accuracy: {acc:.3f}")

# Plot decision boundary
x1_min, x1_max = Xc[:, 0].min() - 0.5, Xc[:, 0].max() + 0.5
x2_min, x2_max = Xc[:, 1].min() - 0.5, Xc[:, 1].max() + 0.5
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 300), np.linspace(x2_min, x2_max, 300))
grid = np.c_[xx1.ravel(), xx2.ravel()]
Z = gda.predict(grid).reshape(xx1.shape)
plt.figure(figsize=(6, 5))
plt.contourf(xx1, xx2, Z, alpha=0.3)
plt.scatter(Xc[:, 0], Xc[:, 1], c=yc, s=10)
plt.xlabel(Xc_cols[0])
plt.ylabel(Xc_cols[1])
plt.title(f"GDA Decision Boundary (acc={acc:.3f})")
plt.savefig("gda_boundary.png", dpi=300)
plt.close()

# ================================
# 5. Linear Regression (Normal Equation)
# ================================
class LinearRegressionNE:
    def __init__(self):
        self.coef_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        Xb = np.c_[np.ones((X.shape[0], 1)), X]
        self.coef_ = np.linalg.pinv(Xb) @ y
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        Xb = np.c_[np.ones((X.shape[0], 1)), X]
        return (Xb @ self.coef_).ravel()

Xr, yr = df_reg[Xr_cols].values, df_reg[yr_col].values.astype(float)
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_ratio=0.2, seed=1126)

linreg = LinearRegressionNE().fit(Xr_tr, yr_tr)
yr_pred = linreg.predict(Xr_te)
mse = np.mean((yr_pred - yr_te) ** 2)
ss_tot = np.sum((yr_te - yr_te.mean()) ** 2)
ss_res = np.sum((yr_pred - yr_te) ** 2)
r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
print(f"Regression MSE: {mse:.3f}, R2: {r2:.3f}")

x1_min, x1_max = Xr[:, 0].min() - 0.5, Xr[:, 0].max() + 0.5
x2_min, x2_max = Xr[:, 1].min() - 0.5, Xr[:, 1].max() + 0.5
gx1, gx2 = np.meshgrid(np.linspace(x1_min, x1_max, 200), np.linspace(x2_min, x2_max, 200))
grid_reg = np.c_[gx1.ravel(), gx2.ravel()]
grid_pred = linreg.predict(grid_reg).reshape(gx1.shape)
plt.figure(figsize=(6, 5))
plt.contourf(gx1, gx2, grid_pred, alpha=0.8)
plt.scatter(Xr[:, 0], Xr[:, 1], c=yr, s=10)
plt.xlabel(Xr_cols[0])
plt.ylabel(Xr_cols[1])
plt.title(f"Linear Regression Surface (MSE={mse:.2f}, R2={r2:.3f})")
plt.colorbar()
plt.savefig("regression_surface.png", dpi=300)
plt.close()

# ================================
# 6. Piecewise Function h(x)
# ================================
def C_predict(X):
    return gda.predict(X)

def R_predict(X):
    return linreg.predict(X)

def h_predict(X):
    c = C_predict(X)
    r = R_predict(X)
    return np.where(c == 1, r, -999.0)

h_out = h_predict(Xr)
h_grid = h_predict(grid_reg).reshape(gx1.shape)

plt.figure(figsize=(6, 5))
plt.contourf(gx1, gx2, h_grid, alpha=0.8)
plt.scatter(Xr[:, 0], Xr[:, 1], c=h_out, s=10)
plt.xlabel(Xr_cols[0])
plt.ylabel(Xr_cols[1])
plt.title("Piecewise Function h(x)")
plt.colorbar()
plt.savefig("piecewise_function.png", dpi=300)
plt.close()

print("All figures saved: gda_boundary.png, regression_surface.png, piecewise_function.png")
