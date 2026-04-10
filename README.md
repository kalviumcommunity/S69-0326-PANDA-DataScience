# 📦 Retail Inventory Analysis & Demand Insights

## 📌 Problem Statement

Retail chains track inventory movement but frequently experience **stockouts** (running out of products) or **overstocking** (excess inventory).

The goal of this project is to analyze inventory and sales data to:

* Identify **slow-moving items**
* Detect **fast-selling products**
* Understand **demand variability across stores and time periods**

---

## 📊 Dataset

This project uses the dataset:
👉 [Retail Store Inventory Forecasting Dataset]([https://www.selectdataset.com/dataset/95a4b1fe05a5892d8e2ad0926859db14?utm_source=chatgpt.com](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset))


* Contains **daily records across multiple stores and products**
* Includes key features like:

  * Sales (Units Sold)
  * Inventory levels
  * Pricing & discounts
  * Store & product details
  * External factors (weather, holidays, promotions)

Such datasets are widely used for **time-series forecasting, inventory optimization, and retail analytics** ([Open Data Bay][2])

---

## 🎯 Objectives

* Analyze product performance (fast vs slow moving)
* Detect stockout and overstock patterns
* Study demand variation across:

  * Stores
  * Regions
  * Time (daily/seasonal trends)
* Provide insights to improve:

  * Inventory planning
  * Demand forecasting
  * Business decision-making

---

## 🧠 Key Analysis Performed

* 📈 Exploratory Data Analysis (EDA)
* 🛒 Product-wise sales trends
* 🏬 Store-wise demand comparison
* 📅 Time series analysis (seasonality & trends)
* ⚖️ Inventory vs sales imbalance detection

---

## 📌 Key Features in Dataset

* `Date` – Daily transaction date
* `Store ID` – Unique store identifier
* `Product ID` – Unique product identifier
* `Category` – Product category
* `Region` – Store location
* `Inventory Level` – Available stock
* `Units Sold` – Sales per day
* `Units Ordered` – Restocking quantity
* `Price` – Product price
* `Discount` – Discount applied
* `Demand Forecast` – Predicted demand
* `Weather / Holiday / Promotion` – External factors affecting demand

---

## 🔍 Insights to Extract

* 📉 Slow-moving products → Low sales, high inventory
* 📈 Fast-selling products → High sales, frequent stockouts
* ⚠️ Overstock situations → High inventory, low turnover
* 🚫 Stockouts → Demand exceeds inventory
* 🌍 Regional demand differences
* 🕒 Seasonal demand patterns

---

## 🛠️ Tech Stack

* Python 🐍
* Pandas & NumPy
* Matplotlib / Seaborn
* Jupyter Notebook

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/your-username/retail-inventory-analysis.git

# Navigate into project folder
cd retail-inventory-analysis

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook
```

---

## 📊 Expected Outcomes

* Better understanding of inventory inefficiencies
* Data-driven decision-making for retail chains
* Improved demand forecasting strategies
* Reduction in stockouts and overstock situations

---




---

## 📜 License

This project is for educational and research purposes.

[1]: https://www.selectdataset.com/dataset/95a4b1fe05a5892d8e2ad0926859db14?utm_source=chatgpt.com "Retail Store Inventory Forecasting Dataset|零售库存管理数据集|需求预测数据集"
[2]: https://www.opendatabay.com/data/consumer/138533b8-da75-419a-b7d7-479fdfec652a?utm_source=chatgpt.com "Inventory Optimisation Dataset CSV Free"
