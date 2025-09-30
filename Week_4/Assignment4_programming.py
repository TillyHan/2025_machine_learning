# Week4_baseline.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
import matplotlib.pyplot as plt

# ============================
# (1) 讀取資料
# ============================
df_cls = pd.read_csv("classification.csv")
df_reg = pd.read_csv("regression.csv")

# ============================
# (2) 分類模型 Logistic Regression
# ============================
Xc = df_cls[["lon", "lat"]].values
yc = df_cls["label"].values

Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
    Xc, yc, test_size=0.2, random_state=42, stratify=yc
)

clf = LogisticRegression(max_iter=1000)
clf.fit(Xc_tr, yc_tr)
yc_pred = clf.predict(Xc_te)

print("=== Classification (Logistic Regression) ===")
print("Accuracy:", accuracy_score(yc_te, yc_pred))
print(classification_report(yc_te, yc_pred))

plt.hist(yc_te, alpha=0.5, label="True", bins=3)
plt.hist(yc_pred, alpha=0.5, label="Pred", bins=3)
plt.xlabel("Label")
plt.ylabel("Count")
plt.title("Classification: Label Distribution (Test vs Pred)")
plt.legend()
plt.savefig("classification_hist.png")
plt.close()

# ============================
# (3) 回歸模型 Linear Regression
# ============================
Xr = df_reg[["lon", "lat"]].values
yr = df_reg["value"].values

Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
    Xr, yr, test_size=0.2, random_state=42
)

regr = LinearRegression()
regr.fit(Xr_tr, yr_tr)
yr_pred = regr.predict(Xr_te)

mse = mean_squared_error(yr_te, yr_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(yr_te, yr_pred)
r2 = r2_score(yr_te, yr_pred)

print("\n=== Regression (Linear Regression) ===")
print("MSE :", mse)
print("RMSE:", rmse)
print("MAE :", mae)
print("R^2 :", r2)

plt.scatter(yr_te, yr_pred, s=8)
plt.xlabel("True Temperature (°C)")
plt.ylabel("Predicted Temperature (°C)")
plt.title("Linear Regression: True vs Predicted")
plt.savefig("regression_true_vs_pred.png")
plt.close()
