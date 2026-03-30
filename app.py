import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from bs4 import BeautifulSoup
import os

st.set_page_config(layout="wide", page_title="RetailLens Dashboard")

# Function to get base64 encoded image from a matplotlib figure
def get_base64_chart(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', transparent=True)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{encoded}"

@st.cache_data
def load_data():
    df = pd.read_csv("retail_store_inventory.csv")
    df = df.drop(columns=['Seasonality','Weather Condition','Holiday/Promotion'], errors='ignore')
    df = df.drop_duplicates()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna()
    df['Stockout'] = df['Units Sold'] >= dict(df)['Inventory Level'] if 'Inventory Level' in df else False
    # Based on main.ipynb:
    df['Stockout'] = df['Units Sold'] >= df['Inventory Level']
    df['Overstock'] = df['Inventory Level'] > df['Units Sold'] * 2
    df['Sales Ratio'] = df['Units Sold'] / df['Inventory Level']
    df['Demand Gap'] = df['Demand Forecast'] - df['Units Sold']
    df['day_of_week'] = df['Date'].dt.dayofweek
    return df

df = load_data()

# ==========================================
# 1. Calculate EXACT KPIs from main.ipynb
# ==========================================
product_sales = df.groupby('Product ID')['Units Sold'].sum().sort_values()
slowest_product = product_sales.index[0]
fastest_product = product_sales.index[-1]

stockout_count = df['Stockout'].sum()
overstock_count = df['Overstock'].sum()

# ==========================================
# 2. Generate EXACT Charts from main.ipynb
# ==========================================
sns.set_theme(style="whitegrid")

# Chart 1: Top 10 Fast Selling Products
fast_products = df.groupby('Product ID')['Sales Ratio'].mean().sort_values(ascending=False).head(10)
fig1, ax1 = plt.subplots(figsize=(6, 4))
fast_products.plot(kind='bar', title="Top 10 Fast Selling Products", ax=ax1, color="green")
ax1.set_ylabel("Mean Sales Ratio")
fig1.tight_layout()
c1 = get_base64_chart(fig1)

# Chart 2: Top 10 Slow Moving Products
slow_products = df.groupby('Product ID')['Sales Ratio'].mean().sort_values().head(10)
fig2, ax2 = plt.subplots(figsize=(6, 4))
slow_products.plot(kind='bar', title="Top 10 Slow Moving Products", ax=ax2, color="red")
ax2.set_ylabel("Mean Sales Ratio")
fig2.tight_layout()
c2 = get_base64_chart(fig2)

# Chart 3: Stockout vs Overstock Count
fig3, ax3 = plt.subplots(figsize=(6, 4))
df[['Stockout','Overstock']].sum().plot(kind='bar', title="Stockout vs Overstock Count", ax=ax3, color=["orange", "blue"])
ax3.set_ylabel("Count")
fig3.tight_layout()
c3 = get_base64_chart(fig3)

# Chart 4: High Demand Variability Products
demand_variability = df.groupby('Product ID')['Units Sold'].std()
fig4, ax4 = plt.subplots(figsize=(6, 4))
demand_variability.sort_values(ascending=False).head(10).plot(kind='bar', title="High Demand Variability Products", ax=ax4, color="purple")
ax4.set_ylabel("Standard Deviation (Units Sold)")
fig4.tight_layout()
c4 = get_base64_chart(fig4)

# Chart 5: Demand Variability per Store
store_variability = df.groupby('Store ID')['Units Sold'].std()
fig5, ax5 = plt.subplots(figsize=(6, 4))
store_variability.plot(kind='bar', title="Demand Variability per Store", ax=ax5, color="teal")
ax5.set_ylabel("Standard Deviation")
fig5.tight_layout()
c5 = get_base64_chart(fig5)

# Chart 6: Average Sales by Day of Week
weekly_sales = df.groupby('day_of_week')['Units Sold'].mean()
fig6, ax6 = plt.subplots(figsize=(6, 4))
weekly_sales.plot(kind='line', title="Average Sales by Day of Week", ax=ax6, marker='o', linewidth=2)
ax6.set_ylabel("Mean Units Sold")
ax6.set_xlabel("Day of Week (0=Mon, 6=Sun)")
fig6.tight_layout()
c6 = get_base64_chart(fig6)

# ==========================================
# 3. Dynamic HTML Construction
# ==========================================
kpi_html = f"""
    <div class="kpi-card">
      <div class="kpi-label">Stockout Events</div>
      <div class="kpi-value red">{stockout_count:,}</div>
      <div class="kpi-sub">Total Units Sold >= Inventory</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Overstock Events</div>
      <div class="kpi-value blue">{overstock_count:,}</div>
      <div class="kpi-sub">Inventory > 2x Units Sold</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Fastest Mover</div>
      <div class="kpi-value">{fastest_product}</div>
      <div class="kpi-sub">Highest Aggregate Units</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Slowest Mover</div>
      <div class="kpi-value">{slowest_product}</div>
      <div class="kpi-sub">Lowest Aggregate Units</div>
    </div>
"""

main_html = f"""
  <section id="section1">
    <div class="section-header">
      <div class="section-num">01</div>
      <div class="section-title-block">
        <h2>Sales Ratio Performance</h2>
        <p>Top 10 Fast and Slow Moving Products based on Average Sales Ratio.</p>
      </div>
    </div>
    <div class="two-col">
      <div class="chart-card"><img src="{c1}"><div class="chart-body"><h3>Fast Selling Products</h3></div></div>
      <div class="chart-card"><img src="{c2}"><div class="chart-body"><h3>Slow Moving Products</h3></div></div>
    </div>
  </section>

  <section id="section2">
    <div class="section-header">
      <div class="section-num">02</div>
      <div class="section-title-block">
        <h2>Inventory Constraints</h2>
        <p>Stockout vs Overstock Counts and High Demand Variability Products.</p>
      </div>
    </div>
    <div class="two-col">
      <div class="chart-card"><img src="{c3}"><div class="chart-body"><h3>Stockout vs Overstock</h3></div></div>
      <div class="chart-card"><img src="{c4}"><div class="chart-body"><h3>Demand Variability by Product</h3></div></div>
    </div>
  </section>

  <section id="section3">
    <div class="section-header">
      <div class="section-num">03</div>
      <div class="section-title-block">
        <h2>Store & Time Trends</h2>
        <p>Demand Variability per Store and Average Sales by Day of Week.</p>
      </div>
    </div>
    <div class="two-col">
      <div class="chart-card"><img src="{c5}"><div class="chart-body"><h3>Store Variability</h3></div></div>
      <div class="chart-card"><img src="{c6}"><div class="chart-body"><h3>Weekly Sales Trend</h3></div></div>
    </div>
  </section>
"""

html_file_path = "website.html"
if os.path.exists(html_file_path):
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Inject new KPIs
    kpi_grid = soup.find('div', class_='kpi-grid')
    if kpi_grid:
        kpi_grid.clear()
        kpi_grid.append(BeautifulSoup(kpi_html, 'html.parser'))
        
    # Inject new Main Content
    main_tag = soup.find('main')
    if main_tag:
        main_tag.clear()
        main_tag.append(BeautifulSoup(main_html, 'html.parser'))
        
    # Render final HTML
    final_html = str(soup)
    st.components.v1.html(final_html, height=2800, scrolling=True)
    
else:
    st.error(f"Cannot find {html_file_path}")
