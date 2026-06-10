import warnings
warnings.filterwarnings("ignore")
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from features import get_feature_matrix

MODELS_DIR = Path("ml/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def sep(t): print(f"\n{'='*55}\n  {t}\n{'='*55}")

def reg_report(yt, yp, label):
    mae  = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    r2   = r2_score(yt, yp)
    mape = np.mean(np.abs((yt - yp) / (yt + 1e-8))) * 100
    print(f"\n  {label}")
    print(f"  MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.4f}  MAPE={mape:.2f}%")
    return {"MAE": round(mae,3), "RMSE": round(rmse,3), "R2": round(r2,4), "MAPE": round(mape,2)}

def clf_report(yt, yp, yprob=None, label=""):
    acc = accuracy_score(yt, yp)
    f1w = f1_score(yt, yp, average="weighted", zero_division=0)
    f1m = f1_score(yt, yp, average="macro",    zero_division=0)
    pre = precision_score(yt, yp, average="weighted", zero_division=0)
    rec = recall_score(yt, yp, average="weighted", zero_division=0)
    print(f"\n  {label}")
    print(f"  Accuracy={acc:.4f}  F1_weighted={f1w:.4f}  F1_macro={f1m:.4f}")
    print(f"  Precision={pre:.4f}  Recall={rec:.4f}")
    if yprob is not None:
        try:
            n = len(np.unique(yt))
            auc = roc_auc_score(yt, yprob[:,1]) if n==2 else roc_auc_score(
                pd.get_dummies(yt).values, yprob, multi_class="ovr", average="weighted")
            print(f"  ROC-AUC={auc:.4f}")
        except: pass
    print(classification_report(yt, yp, zero_division=0))
    return {"Accuracy": round(acc,4), "F1_weighted": round(f1w,4),
            "F1_macro": round(f1m,4), "Precision": round(pre,4), "Recall": round(rec,4)}

sep("Loading data")
df = pd.read_csv("ml/data/retail_dataset.csv", parse_dates=["timestamp"])
df = df.dropna(subset=["price_tier","high_demand","units_sold","revenue"])
print(f"  Rows: {len(df)}")
X = get_feature_matrix(df)

sep("Task 1: Sales Forecast — XGBoost")
y1 = df["units_sold"].values
Xt1, Xe1, yt1, ye1 = train_test_split(X, y1, test_size=0.2, random_state=42)
m1 = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0, n_jobs=-1)
m1.fit(Xt1, yt1)
r1 = reg_report(ye1, m1.predict(Xe1), "XGBoost → units_sold")
cv1 = cross_val_score(m1, Xt1, yt1, cv=KFold(5,shuffle=True,random_state=42), scoring="r2")
print(f"  CV R2: {cv1.mean():.4f} ± {cv1.std():.4f}")
joblib.dump(m1, MODELS_DIR/"xgb_sales.pkl")
print("  Saved: ml/models/xgb_sales.pkl")

sep("Task 2: Revenue Forecast — LightGBM")
y2 = df["revenue"].values
Xt2, Xe2, yt2, ye2 = train_test_split(X, y2, test_size=0.2, random_state=42)
m2 = LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=-1)
m2.fit(Xt2, yt2)
r2m = reg_report(ye2, m2.predict(Xe2), "LightGBM → revenue")
cv2 = cross_val_score(m2, Xt2, yt2, cv=KFold(5,shuffle=True,random_state=42), scoring="r2")
print(f"  CV R2: {cv2.mean():.4f} ± {cv2.std():.4f}")
joblib.dump(m2, MODELS_DIR/"lgbm_revenue.pkl")
print("  Saved: ml/models/lgbm_revenue.pkl")

sep("Task 3: Price Tier — Random Forest")
le = LabelEncoder()
y3 = le.fit_transform(df["price_tier"])
Xt3, Xe3, yt3, ye3 = train_test_split(X, y3, test_size=0.2, random_state=42, stratify=y3)
m3 = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5,
                             class_weight="balanced", random_state=42, n_jobs=-1)
m3.fit(Xt3, yt3)
r3 = clf_report(ye3, m3.predict(Xe3), m3.predict_proba(Xe3), "RandomForest → price_tier")
cv3 = cross_val_score(m3, Xt3, yt3,
                      cv=StratifiedKFold(5,shuffle=True,random_state=42), scoring="f1_weighted")
print(f"  CV F1: {cv3.mean():.4f} ± {cv3.std():.4f}")
joblib.dump(m3, MODELS_DIR/"rf_tier.pkl")
joblib.dump(le, MODELS_DIR/"le_tier.pkl")
print("  Saved: ml/models/rf_tier.pkl")

sep("Task 4: High Demand — Gradient Boosting")
y4 = df["high_demand"].values
Xt4, Xe4, yt4, ye4 = train_test_split(X, y4, test_size=0.2, random_state=42, stratify=y4)
m4 = GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                 learning_rate=0.05, subsample=0.8, random_state=42)
m4.fit(Xt4, yt4)
r4 = clf_report(ye4, m4.predict(Xe4), m4.predict_proba(Xe4), "GradBoost → high_demand")
cv4 = cross_val_score(m4, Xt4, yt4,
                      cv=StratifiedKFold(5,shuffle=True,random_state=42), scoring="f1")
print(f"  CV F1: {cv4.mean():.4f} ± {cv4.std():.4f}")
joblib.dump(m4, MODELS_DIR/"gb_demand.pkl")
print("  Saved: ml/models/gb_demand.pkl")

sep("ALL MODELS TRAINED AND SAVED")
print("  Run app.py next: python app.py")