from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_jwt_extended import (
    JWTManager, create_access_token,
    jwt_required, get_jwt_identity
)
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3, joblib, os, sys
import pandas as pd
import numpy as np
from datetime import timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ml"))
from features import get_feature_matrix

app = Flask(__name__)
app.config["JWT_SECRET_KEY"] = "pricing_secret_key_2024"
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=24)
jwt = JWTManager(app)

DB = "users.db"
MODELS = {}

def get_db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            base_price REAL, selling_price REAL, category TEXT,
            discount_pct REAL, demand_index REAL,
            pred_units INTEGER, pred_revenue REAL,
            pred_tier TEXT, pred_high_demand INTEGER, pred_demand_prob REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()

def load_models():
    base = os.path.join(os.path.dirname(__file__), "ml", "models")
    try:
        MODELS["sales"]   = joblib.load(os.path.join(base, "xgb_sales.pkl"))
        MODELS["revenue"] = joblib.load(os.path.join(base, "lgbm_revenue.pkl"))
        MODELS["tier"]    = joblib.load(os.path.join(base, "rf_tier.pkl"))
        MODELS["le_tier"] = joblib.load(os.path.join(base, "le_tier.pkl"))
        MODELS["demand"]  = joblib.load(os.path.join(base, "gb_demand.pkl"))
        print("All models loaded successfully!")
        return True
    except FileNotFoundError as e:
        print(f"Models not found: {e}")
        print("Please run: python ml/train.py first")
        return False

# ── Page routes ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return redirect(url_for("login_page"))

@app.route("/login")
def login_page():
    return render_template("login.html")

@app.route("/register")
def register_page():
    return render_template("register.html")

@app.route("/dashboard")
def dashboard_page():
    return render_template("dashboard.html")

# ── Auth API ──────────────────────────────────────────────────────────────────
@app.route("/api/register", methods=["POST"])
def register():
    data = request.get_json()
    username = data.get("username", "").strip()
    email    = data.get("email", "").strip()
    password = data.get("password", "")
    if not username or not email or not password:
        return jsonify({"error": "All fields are required"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400
    hashed = generate_password_hash(password)
    try:
        conn = get_db()
        conn.execute("INSERT INTO users (username, email, password) VALUES (?,?,?)",
                     (username, email, hashed))
        conn.commit()
        conn.close()
        return jsonify({"message": "Account created! Please login."}), 201
    except sqlite3.IntegrityError:
        return jsonify({"error": "Username or email already exists"}), 409

@app.route("/api/login", methods=["POST"])
def login():
    data     = request.get_json()
    email    = data.get("email", "").strip()
    password = data.get("password", "")
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    conn.close()
    if not user or not check_password_hash(user["password"], password):
        return jsonify({"error": "Invalid email or password"}), 401
    token = create_access_token(identity=str(user["id"]))
    return jsonify({
        "token": token,
        "username": user["username"],
        "message": "Login successful"
    }), 200

# ── Predict API ───────────────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
@jwt_required()
def predict():
    if not MODELS:
        return jsonify({"error": "Models not loaded. Run ml/train.py first."}), 503

    user_id = int(get_jwt_identity())
    data    = request.get_json()

    try:
        base_price       = float(data["base_price"])
        selling_price    = float(data["selling_price"])
        cost_price       = float(data.get("cost_price", base_price * 0.5))
        competitor_price = float(data.get("competitor_price", base_price * 1.0))
        stock_level      = int(data.get("stock_level", 100))
        product_age_days = int(data.get("product_age_days", 180))
        demand_index     = float(data.get("demand_index", 1.0))
        discount_pct     = float(data.get("discount_pct", 0))
        customer_rating  = float(data.get("customer_rating", 4.0))
        reviews_count    = int(data.get("reviews_count", 100))
        category         = data.get("category", "Electronics")
        promo_active     = int(data.get("promo_active", 0))
        is_weekend       = int(data.get("is_weekend", 0))
        holiday_season   = int(data.get("holiday_season", 0))
        margin_ratio     = round((selling_price - cost_price) / max(selling_price, 0.01), 4)

        record = {
            "base_price": base_price, "cost_price": cost_price,
            "selling_price": selling_price, "competitor_price": competitor_price,
            "stock_level": stock_level, "product_age_days": product_age_days,
            "demand_index": demand_index, "discount_pct": discount_pct,
            "customer_rating": customer_rating, "reviews_count": reviews_count,
            "margin_ratio": margin_ratio, "category": category,
            "is_weekend": is_weekend, "promo_active": promo_active,
            "holiday_season": holiday_season, "day_of_week": 1,
            "month": 6, "hour": 12, "quarter": 2,
        }
        X = get_feature_matrix(pd.DataFrame([record]))

        units   = int(round(MODELS["sales"].predict(X)[0]))
        revenue = round(float(MODELS["revenue"].predict(X)[0]), 2)
        tier_e  = MODELS["tier"].predict(X)[0]
        tier    = MODELS["le_tier"].inverse_transform([tier_e])[0]
        demand  = int(MODELS["demand"].predict(X)[0])
        d_prob  = round(float(MODELS["demand"].predict_proba(X)[0][1]), 4)

        conn = get_db()
        conn.execute("""
            INSERT INTO predictions
            (user_id,base_price,selling_price,category,discount_pct,demand_index,
             pred_units,pred_revenue,pred_tier,pred_high_demand,pred_demand_prob)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (user_id, base_price, selling_price, category, discount_pct,
              demand_index, units, revenue, tier, demand, d_prob))
        conn.commit()
        conn.close()

        return jsonify({
            "units_sold"      : units,
            "revenue"         : revenue,
            "price_tier"      : tier,
            "high_demand"     : demand,
            "demand_prob"     : d_prob,
            "margin_ratio"    : round(margin_ratio * 100, 1),
        }), 200

    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400

# ── History API ───────────────────────────────────────────────────────────────
@app.route("/api/history", methods=["GET"])
@jwt_required()
def history():
    user_id = int(get_jwt_identity())
    conn    = get_db()
    rows    = conn.execute("""
        SELECT * FROM predictions WHERE user_id=?
        ORDER BY created_at DESC LIMIT 20
    """, (user_id,)).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows]), 200

if __name__ == "__main__":
    init_db()
    models_ok = load_models()
    if not models_ok:
        print("\nWARNING: Models missing. Train them first with: python ml/train.py")
    app.run(debug=True, port=5000)