# eda_and_prepare.py
import os, pandas as pd, matplotlib.pyplot as plt, numpy as np
BASE = os.path.dirname(__file__)
CSV = os.path.join(BASE, "Supermart Grocery Sales - Retail Analytics Dataset.csv")
OUTDIR = os.path.join(BASE, "static", "charts")
os.makedirs(OUTDIR, exist_ok=True)

def load_or_create():
    if os.path.exists(CSV):
        df = pd.read_csv(CSV, encoding='utf-8', low_memory=False)
    else:
        raise SystemExit("CSV not found. Run create_dummy_supermart.py or place the CSV file.")
    return df

def clean(df):
    # lowercase columns
    df.columns = [c.strip().lower().replace(' ','_') for c in df.columns]
    # try to parse dates
    for c in df.columns:
        if 'date' in c:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    # numeric conversions
    for c in df.select_dtypes(include=['object']).columns:
        if df[c].str.replace(',','').str.replace('.','',1).str.isnumeric().any():
            try:
                df[c] = pd.to_numeric(df[c].str.replace(',',''), errors='coerce')
            except:
                pass
    # fill common NaNs sensibly
    df['total'] = df.get('total', None)
    if 'quantity' in df.columns and 'unit_price' in df.columns and (df['total'].isna().sum()>0):
        df['total'] = df['quantity'] * df['unit_price']
    # drop duplicates and obvious junk
    df = df.drop_duplicates().reset_index(drop=True)
    return df

def create_charts(df):
    # sales by store / city
    if 'store_location' in df.columns:
        s = df.groupby('store_location')['total'].sum().sort_values(ascending=False).head(12)
        plt.figure(figsize=(8,4)); s.plot.bar(); plt.title('Top store locations by sales'); plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR,'sales_by_location.png')); plt.close()
    # top products
    if 'product' in df.columns:
        p = df.groupby('product')['quantity'].sum().sort_values(ascending=False).head(12)
        plt.figure(figsize=(8,4)); p.plot.bar(); plt.title('Top products by quantity'); plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR,'top_products.png')); plt.close()
    # sales by month
    datecol = None
    for c in df.columns:
        if 'date' in c:
            datecol = c; break
    if datecol:
        tmp = df.copy(); tmp['month']=tmp[datecol].dt.to_period('M')
        m = tmp.groupby('month')['total'].sum().sort_index()
        plt.figure(figsize=(10,4)); m.plot(marker='o'); plt.title('Sales by month'); plt.xticks(rotation=45); plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR,'sales_by_month.png')); plt.close()
    # correlation heatmap if numeric cols
    num = df.select_dtypes(include=[np.number]).corr()
    if not num.empty:
        plt.figure(figsize=(6,5)); import seaborn as sns
        sns.heatmap(num, annot=False, cmap='coolwarm'); plt.title('Numeric correlation'); plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR,'corr_heatmap.png')); plt.close()

def main():
    df = load_or_create()
    df = clean(df)
    df.to_csv(os.path.join(BASE,'supermart_cleaned.csv'), index=False)
    create_charts(df)
    print("Saved cleaned CSV and charts to static/charts")

if __name__ == '__main__':
    main()
