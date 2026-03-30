import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("retail_store_inventory.csv")
df.head()

df.info()

df=df.drop(columns=['Seasonality','Weather Condition','Holiday/Promotion'])

df.info()

df = df.drop_duplicates()


df['Date'] = pd.to_datetime(df['Date'])

df.dtypes


df.head()

df = df.dropna()

df.describe()

df.head()

product_sales = df.groupby('Product ID')['Units Sold'].sum().sort_values()

slow_products = product_sales.head(10)
fast_products = product_sales.tail(10)

print("Slow Moving Products:\n", slow_products)
print("\nFast Moving Products:\n", fast_products)

df['Stockout'] = df['Units Sold'] > df['Inventory Level']

stockout_count = df['Stockout'].value_counts()
print(stockout_count)

df['Overstock'] = df['Inventory Level'] > df['Units Sold'] * 2

overstock_count = df['Overstock'].value_counts()
print(overstock_count)

df['Sales Ratio'] = df['Units Sold'] / df['Inventory Level']

df['Demand Gap'] = df['Demand Forecast'] - df['Units Sold']

fast_products = df.groupby('Product ID')['Sales Ratio'].mean().sort_values(ascending=False).head(10)
fast_products.plot(kind='bar', title="Top 10 Fast Selling Products")
plt.show()

slow_products = df.groupby('Product ID')['Sales Ratio'].mean().sort_values().head(10)
slow_products.plot(kind='bar', title="Top 10 Slow Moving Products")
plt.show()

df['Stockout'] = df['Units Sold'] >= df['Inventory Level']
df['Overstock'] = df['Inventory Level'] > df['Units Sold'] * 2

df[['Stockout','Overstock']].sum().plot(kind='bar')
plt.title("Stockout vs Overstock Count")
plt.show()

demand_variability = df.groupby('Product ID')['Units Sold'].std()

demand_variability.sort_values(ascending=False).head(10).plot(kind='bar')
plt.title("High Demand Variability Products")
plt.show()

store_variability = df.groupby('Store ID')['Units Sold'].std()

store_variability.plot(kind='bar', title="Demand Variability per Store")
plt.show()

df['day_of_week'] = df['Date'].dt.dayofweek

weekly_sales = df.groupby('day_of_week')['Units Sold'].mean()

weekly_sales.plot(kind='line', title="Average Sales by Day of Week")
plt.show()

