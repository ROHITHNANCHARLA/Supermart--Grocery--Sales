# create_dummy_supermart.py
import pandas as pd, random, os
dates = pd.date_range("2022-01-01","2023-12-31", freq='D')
products = ['Milk','Bread','Eggs','Butter','Apples','Bananas','Rice','Sugar','Soap','Toothpaste','Shampoo']
stores = ['Downtown','Mall','Uptown','Suburb','Airport']
rows=[]
random.seed(42)
for d in dates:
    for s in stores:
        for p in random.sample(products, k=5):
            qty = random.randint(1,30)
            price = round(random.uniform(0.5,10.0),2)
            rows.append({'date':d.strftime('%Y-%m-%d'),'store_location':s,'product':p,'quantity':qty,'unit_price':price,'total':qty*price})
df = pd.DataFrame(rows)
df.to_csv('Supermart Grocery Sales - Retail Analytics Dataset.csv', index=False)
print("Created synthetic dataset")
