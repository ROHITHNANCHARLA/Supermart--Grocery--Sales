# model_build.py
import os, joblib, json, sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(__file__)
CSV = os.path.join(BASE,'supermart_cleaned.csv')
MODEL = os.path.join(BASE,'model_supermart.pkl')
FEATURES = os.path.join(BASE,'feature_columns.json')
DB = os.path.join(BASE,'supermart.db')

if not os.path.exists(CSV):
    raise SystemExit("Run eda_and_prepare.py first to create supermart_cleaned.csv")

df = pd.read_csv(CSV, low_memory=False)
# choose target
target = 'total' if 'total' in df.columns else ('quantity' if 'quantity' in df.columns else None)
if target is None:
    raise SystemExit("No suitable target (total/quantity) found in cleaned CSV.")

# basic feature engineering
X = pd.DataFrame()
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    X['year'] = df['date'].dt.year.fillna(0)
    X['month'] = df['date'].dt.month.fillna(0)
# create dummies for product and store_location top categories
for col in ['product','store_location','category']:
    if col in df.columns:
        top = df[col].value_counts().nlargest(20).index.tolist()
        X[col+'_top'] = df[col].where(df[col].isin(top),'other')
if 'quantity' in df.columns:
    X['quantity'] = df['quantity']
if 'unit_price' in df.columns:
    X['unit_price'] = df['unit_price']

X = pd.get_dummies(X, drop_first=True)
y = df[target].astype(float).fillna(0.0)
mask = y.notna() & X.notna().all(axis=1)
X = X[mask]; y = y[mask]

# persist feature columns
json.dump(X.columns.tolist(), open(FEATURES,'w'), indent=2)

pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler()), ('rf', RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1))])
if len(X) < 10:
    print("Not enough rows to train. Need at least 10 rows.")
else:
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    mae = (abs(preds - y_test)).mean()
    joblib.dump(pipe, MODEL)
    print("Trained model saved to", MODEL, "MAE:", mae)

# create sqlite db with raw and aggregated tables for SQL page
conn = sqlite3.connect(DB)
df.to_sql('supermart_raw', conn, if_exists='replace', index=False)
# aggregated monthly sales
if 'date' in df.columns:
    agg = df.groupby([df['date'].dt.to_period('M')]).agg({'total':'sum','quantity':'sum'}).reset_index()
    agg['month'] = agg['date'].astype(str)
    agg[['month','total','quantity']].to_sql('sales_by_month', conn, if_exists='replace', index=False)
conn.close()
print("DB built at", DB)
