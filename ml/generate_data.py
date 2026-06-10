import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

def generate_dataset(n_samples=10000):
    dates = pd.date_range(start="2022-01-01", periods=n_samples, freq="h")
    df = pd.DataFrame({"timestamp": dates})
    df["day_of_week"]   = df["timestamp"].dt.dayofweek
    df["month"]         = df["timestamp"].dt.month
    df["hour"]          = df["timestamp"].dt.hour
    df["is_weekend"]    = (df["day_of_week"] >= 5).astype(int)
    df["quarter"]       = df["timestamp"].dt.quarter

    categories = ["Electronics", "Clothing", "Food", "Home", "Sports"]
    df["category"]         = np.random.choice(categories, n_samples)
    df["base_price"]       = np.random.uniform(10, 500, n_samples).round(2)
    df["cost_price"]       = (df["base_price"] * np.random.uniform(0.4, 0.7, n_samples)).round(2)
    df["stock_level"]      = np.random.randint(0, 500, n_samples)
    df["product_age_days"] = np.random.randint(1, 730, n_samples)
    df["competitor_price"] = (df["base_price"] * np.random.uniform(0.85, 1.15, n_samples)).round(2)
    df["demand_index"]     = np.random.uniform(0.5, 2.0, n_samples)
    df["promo_active"]     = np.random.choice([0, 1], n_samples, p=[0.75, 0.25])
    df["discount_pct"]     = np.where(df["promo_active"] == 1,
                                       np.random.uniform(5, 40, n_samples), 0).round(1)
    df["customer_rating"]  = np.random.uniform(1.0, 5.0, n_samples).round(1)
    df["reviews_count"]    = np.random.randint(0, 5000, n_samples)
    df["holiday_season"]   = df["month"].isin([11, 12, 1]).astype(int)
    df["selling_price"]    = (
        df["base_price"] * (1 - df["discount_pct"] / 100)
        * np.random.uniform(0.95, 1.05, n_samples)
    ).round(2)

    units_raw = (
        20
        + (-0.3 * (df["selling_price"] / df["base_price"])) * 15
        + 0.5 * df["demand_index"] * 10
        + 0.4 * df["promo_active"] * 12
        + 0.2 * df["holiday_season"] * 8
        + 0.1 * df["customer_rating"] * 3
        + 0.15 * df["is_weekend"] * 5
        + np.random.normal(0, 3, n_samples)
    )
    df["units_sold"]   = np.clip(units_raw, 0, 200).round().astype(int)
    df["revenue"]      = (df["selling_price"] * df["units_sold"]).round(2)
    df["margin_ratio"] = ((df["selling_price"] - df["cost_price"]) / df["selling_price"]).round(4)
    df["price_tier"]   = pd.cut(df["margin_ratio"],
                                bins=[-np.inf, 0.2, 0.45, np.inf],
                                labels=["Low", "Medium", "High"])
    df["high_demand"]  = (df["units_sold"] >= df["units_sold"].quantile(0.70)).astype(int)
    return df

if __name__ == "__main__":
    Path("ml/data").mkdir(parents=True, exist_ok=True)
    df = generate_dataset(10000)
    df.to_csv("ml/data/retail_dataset.csv", index=False)
    print(f"Dataset saved! Shape: {df.shape}")