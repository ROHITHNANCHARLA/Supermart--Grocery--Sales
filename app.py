import os, sqlite3, json, joblib, datetime
from flask import Flask, render_template, request, flash, Response, send_from_directory, redirect, url_for, jsonify
import pandas as pd
import matplotlib.pyplot as plt

BASE = os.path.dirname(__file__)
DB = os.path.join(BASE, 'supermart.db')
MODEL = os.path.join(BASE, 'model_supermart.pkl')
FEATURES = os.path.join(BASE, 'feature_columns.json')
OUTPUTS = os.path.join(BASE, 'outputs')
STATIC_CHARTS = os.path.join(BASE, 'static', 'charts')

os.makedirs(OUTPUTS, exist_ok=True)
os.makedirs(STATIC_CHARTS, exist_ok=True)

app = Flask(__name__)
app.secret_key = "supermart_secret_v3"

# ------------------------------
# Ensure predictions table exists
# ------------------------------
def ensure_predictions_table():
    conn = sqlite3.connect(DB)
    # Create table if not exists
    conn.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        product TEXT,
        store_location TEXT,
        date TEXT,
        year INTEGER,
        quantity REAL,
        unit_price REAL,
        predicted_total REAL
    );
    """)

    # ✅ Try to add 'year' column if it doesn’t exist already
    try:
        conn.execute("ALTER TABLE predictions ADD COLUMN year INTEGER;")
    except sqlite3.OperationalError:
        # column already exists — ignore the error
        pass

    conn.commit()
    conn.close()


ensure_predictions_table()

# ------------------------------
# Load Model
# ------------------------------
def load_model():
    if os.path.exists(MODEL):
        try:
            return joblib.load(MODEL)
        except Exception as e:
            print("Model load failed:", e)
    return None

model = load_model()

# ------------------------------
# Build Feature Vector
# ------------------------------
def get_feature_vector(form):
    cols = []
    if os.path.exists(FEATURES):
        try:
            with open(FEATURES,'r') as f:
                cols = json.load(f)
        except:
            cols = []
    row = dict.fromkeys(cols, 0.0)
    product = form.get('product') or ''
    store_location = form.get('store_location') or ''
    date = form.get('date') or ''
    try:
        year = pd.to_datetime(date).year
        month = pd.to_datetime(date).month
    except:
        year = 0; month = 0

    if 'year' in row: row['year'] = year
    if 'month' in row: row['month'] = month
    if 'quantity' in row: row['quantity'] = float(form.get('quantity') or 0)
    if 'unit_price' in row: row['unit_price'] = float(form.get('unit_price') or 0)

    # Dummies
    prod_key = 'product_top_' + product
    if prod_key in row: row[prod_key] = 1.0
    store_key = 'store_location_top_' + store_location
    if store_key in row: row[store_key] = 1.0

    X = pd.DataFrame([row], columns=cols).fillna(0.0) if cols else pd.DataFrame([[year, month, form.get('quantity'), form.get('unit_price')]], columns=['year','month','quantity','unit_price'])
    return X, (product, store_location, date, year)

# ------------------------------
# Home Page
# ------------------------------
@app.route('/')
def index():
    charts = []
    if os.path.exists(STATIC_CHARTS):
        for fname in sorted(os.listdir(STATIC_CHARTS)):
            if fname.endswith('.png'):
                charts.append('charts/' + fname)
    total_rows = 0
    if os.path.exists(DB):
        conn = sqlite3.connect(DB)
        try:
            total_rows = conn.execute("SELECT COUNT(*) FROM supermart_raw").fetchone()[0]
        except:
            total_rows = 0
        conn.close()
    return render_template('index.html', charts=charts, total=total_rows, model_loaded=(model is not None))

# ------------------------------
# Predict Page
# ------------------------------
@app.route('/predict', methods=['GET','POST'])
def predict():
    prediction = None; saved=False
    if request.method == 'POST':
        X, meta = get_feature_vector(request.form)
        product, store_location, date, year = meta
        qty = float(request.form.get('quantity') or 0)
        up = float(request.form.get('unit_price') or 0)
        if model is None:
            flash("Model not available. Run model_build.py to create model_supermart.pkl", "error")
            return redirect(url_for('predict'))
        try:
            pred = model.predict(X)[0]
            prediction = round(float(pred), 2)
        except Exception as e:
            flash("Prediction failed: "+str(e), "error")
            return redirect(url_for('predict'))
        # Save prediction
        try:
            conn = sqlite3.connect(DB)
            conn.execute("""
                INSERT INTO predictions (timestamp, product, store_location, date, year, quantity, unit_price, predicted_total)
                VALUES (?,?,?,?,?,?,?,?)
            """, (datetime.datetime.utcnow().isoformat(), product, store_location, date, year, qty, up, prediction))
            conn.commit(); conn.close()
            flash("✅ Prediction saved successfully.", "success")
            try:
                generate_and_save_charts(product if product else store_location)
            except Exception as e:
                print("Chart generation error:", e)
            return redirect(url_for('predictions'))
        except Exception as e:
            flash("Saving failed: "+str(e), "error")
            return redirect(url_for('predict'))

    # Dropdown data
    stores=[]; products=[]
    if os.path.exists(DB):
        conn = sqlite3.connect(DB)
        try:
            stores = [r[0] for r in conn.execute("SELECT DISTINCT store_location FROM supermart_raw WHERE store_location IS NOT NULL LIMIT 200").fetchall()]
            products = [r[0] for r in conn.execute("SELECT DISTINCT product FROM supermart_raw WHERE product IS NOT NULL LIMIT 200").fetchall()]
        except Exception:
            stores=[]; products=[]
        conn.close()
    return render_template('predict.html', prediction=prediction, stores=stores, products=products, model_loaded=(model is not None))

# ------------------------------
# Predictions Page
# ------------------------------
@app.route('/predictions')
def predictions():
    conn = sqlite3.connect(DB)
    try:
        df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC LIMIT 500;", conn)
    except Exception as e:
        print("SQL read failed:", e)
        df = pd.DataFrame()
    conn.close()
    records = df.to_dict(orient='records') if not df.empty else []
    chart_groups = {}
    if os.path.exists(OUTPUTS):
        for f in sorted(os.listdir(OUTPUTS), reverse=True):
            if f.endswith('.png'):
                key = f.split('_')[0]
                chart_groups.setdefault(key, []).append('outputs/' + f)
    return render_template('predictions.html', records=records, chart_groups=chart_groups)

# ------------------------------
# Download CSV
# ------------------------------
@app.route('/download_predictions')
def download_predictions():
    conn = sqlite3.connect(DB)
    try:
        df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC;", conn)
    except:
        df = pd.DataFrame()
    conn.close()
    if df.empty:
        flash("No predictions to download.", "error")
        return redirect(url_for('predictions'))
    csv = df.to_csv(index=False)
    return Response(csv, mimetype='text/csv', headers={'Content-Disposition':'attachment; filename=supermart_predictions.csv'})

# ------------------------------
# SQL Explorer (Interactive)
# ------------------------------
@app.route('/sql')
def sql_page():
    conn = sqlite3.connect(DB)
    try:
        # 🏪 Top Stores by Total Sales
        df1 = pd.read_sql_query("""
            SELECT store_location AS Store, ROUND(SUM(total),2) AS Total_Sales
            FROM supermart_raw
            GROUP BY store_location
            ORDER BY Total_Sales DESC LIMIT 20;
        """, conn)
    except Exception as e:
        print("SQL df1 error:", e)
        df1 = pd.DataFrame()

    try:
        # 📅 Monthly Sales Trend (latest year)
        df2 = pd.read_sql_query("""
            SELECT strftime('%Y', date) AS Year, strftime('%m', date) AS Month,
                   ROUND(SUM(total),2) AS Total_Sales
            FROM supermart_raw
            GROUP BY Year, Month
            ORDER BY Year DESC, Month ASC;
        """, conn)
    except Exception as e:
        print("SQL df2 error:", e)
        df2 = pd.DataFrame()

    try:
        # 🧾 Top 15 Products by Total Sales
        df3 = pd.read_sql_query("""
            SELECT product AS Product, ROUND(SUM(total),2) AS Total_Sales, SUM(quantity) AS Total_Quantity
            FROM supermart_raw
            GROUP BY product
            ORDER BY Total_Sales DESC LIMIT 15;
        """, conn)
    except Exception as e:
        print("SQL df3 error:", e)
        df3 = pd.DataFrame()

    try:
        # 💰 Average Unit Price by Store
        df4 = pd.read_sql_query("""
            SELECT store_location AS Store, ROUND(AVG(unit_price),2) AS Avg_Unit_Price
            FROM supermart_raw
            GROUP BY store_location
            ORDER BY Avg_Unit_Price DESC LIMIT 15;
        """, conn)
    except Exception as e:
        print("SQL df4 error:", e)
        df4 = pd.DataFrame()


    conn.close()

    return render_template(
        'sql.html',
        df1=df1.to_dict(orient='records'),
        df2=df2.to_dict(orient='records'),
        df3=df3.to_dict(orient='records'),
        df4=df4.to_dict(orient='records')
    )
    # return render_template('sql.html')

@app.route('/filter_data', methods=['POST'])
def filter_data():
    data = request.get_json(force=True)
    year = data.get("year", "")
    product = data.get("product", "")
    store = data.get("store", "")
    conn = sqlite3.connect(DB)
    where = ["1=1"]
    params = []
    if year:
        where.append("strftime('%Y', date)=?")
        params.append(year)
    if product:
        where.append("product=?")
        params.append(product)
    if store:
        where.append("store_location=?")
        params.append(store)
    where_sql = " AND ".join(where)
    try:
        df = pd.read_sql_query(f"SELECT * FROM supermart_raw WHERE {where_sql} LIMIT 2000;", conn, params=params)
        table = df.fillna("").to_dict(orient="records")
        kpi = pd.read_sql_query(f"SELECT SUM(total) as total_sales, SUM(quantity) as total_items, COUNT(*) as transactions, AVG(total) as avg_bill FROM supermart_raw WHERE {where_sql};", conn, params=params).iloc[0].to_dict()
        return jsonify({"table": table, "kpis": kpi})
    except Exception as e:
        return jsonify({"error": str(e)})
    finally:
        conn.close()

# ------------------------------
# Serve Output Charts
# ------------------------------
@app.route('/outputs/<path:filename>')
def outputs(filename):
    filename = filename.replace("\\","/")
    safe = os.path.basename(filename)
    return send_from_directory(OUTPUTS, safe)

# ------------------------------
# Chart Generation
# ------------------------------
def generate_and_save_charts(key):
    conn = sqlite3.connect(DB)
    df_raw = pd.read_sql_query("SELECT * FROM supermart_raw", conn)
    conn.close()

    plt.figure(figsize=(8,4))
    df_raw['date'] = pd.to_datetime(df_raw['date'], errors='coerce')
    df_raw['month'] = df_raw['date'].dt.to_period('M')
    agg = df_raw.groupby('month')['total'].sum()
    agg.plot(marker='o')
    plt.title(f"Monthly Sales Trend - {key}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS, f"{key}_trend.png"))
    plt.close()

if __name__ == "__main__":
    print("🚀 Starting Supermart Flask App")
    app.run(debug=True)
