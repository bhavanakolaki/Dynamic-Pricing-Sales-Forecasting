import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

NUMERIC_FEATURES = [
    "base_price","cost_price","selling_price","competitor_price",
    "stock_level","product_age_days","demand_index","discount_pct",
    "customer_rating","reviews_count","margin_ratio",
]
CATEGORICAL_FEATURES = ["category"]
BINARY_FEATURES = [
    "is_weekend","promo_active","holiday_season",
    "day_of_week","month","hour","quarter",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

def engineer_features(df):
    df = df.copy()
    df["price_vs_competitor"]   = ((df["selling_price"] - df["competitor_price"]) / df["competitor_price"]).round(4)
    df["price_reduction_ratio"] = ((df["base_price"] - df["selling_price"]) / df["base_price"]).round(4)
    df["low_stock_flag"]        = (df["stock_level"] < 20).astype(int)
    df["review_score"]          = (df["customer_rating"] * np.log1p(df["reviews_count"])).round(4)
    df["abs_margin"]            = (df["selling_price"] - df["cost_price"]).round(2)
    return df

def get_feature_matrix(df):
    df = engineer_features(df)
    derived = ["price_vs_competitor","price_reduction_ratio",
               "low_stock_flag","review_score","abs_margin"]
    feature_cols = [c for c in ALL_FEATURES + derived if c in df.columns]
    df_feat = df[feature_cols].copy()
    for col in CATEGORICAL_FEATURES:
        if col in df_feat.columns:
            le = LabelEncoder()
            df_feat[col] = le.fit_transform(df_feat[col].astype(str))
    return df_feat