import os
import time
import pandas as pd
from dotenv import load_dotenv
from google import genai

# ==========================================
# 1. LOAD API KEY
# ==========================================

load_dotenv(dotenv_path=".env")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("API key not found!")
    exit()

# ==========================================
# 2. LOAD CLEANED DATASET
# ==========================================

df = pd.read_csv("data/cleaned_sales.csv")

# Convert Order_Date to proper date format
df["Order_Date"] = pd.to_datetime(
    df["Order_Date"],
    dayfirst=True
)

print("Dataset loaded successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# ==========================================
# 3. BUSINESS METRICS
# ==========================================

total_sales = df["Sales"].sum()

average_sales = df["Sales"].mean()

total_quantity = df["Quantity"].sum()

# ==========================================
# 4. REGION-WISE SALES
# ==========================================

region_sales = (
    df.groupby("Region")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

# ==========================================
# 5. CATEGORY-WISE SALES
# ==========================================

category_sales = (
    df.groupby("Category")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

# ==========================================
# 6. TOP 10 PRODUCTS
# ==========================================

top_products = (
    df.groupby("Product")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

# ==========================================
# 7. MONTHLY SALES
# ==========================================

monthly_sales = (
    df.groupby(
        df["Order_Date"].dt.to_period("M")
    )["Sales"]
    .sum()
)

# ==========================================
# 8. SALES ANOMALY DETECTION
# ==========================================

Q1 = df["Sales"].quantile(0.25)

Q3 = df["Sales"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR

upper_limit = Q3 + 1.5 * IQR

sales_anomalies = df[
    (df["Sales"] < lower_limit)
    |
    (df["Sales"] > upper_limit)
]

# ==========================================
# 9. CREATE DATA SUMMARY FOR GEMINI
# ==========================================

report_data = f"""

DATASET INFORMATION
-------------------
Rows: {df.shape[0]}
Columns: {df.shape[1]}

BUSINESS METRICS
----------------
Total Sales: {total_sales:,.2f}
Average Sales per Order: {average_sales:,.2f}
Total Quantity Sold: {total_quantity:,}

REGION-WISE SALES
-----------------
{region_sales.to_string()}

CATEGORY-WISE SALES
-------------------
{category_sales.to_string()}

TOP 10 PRODUCTS
---------------
{top_products.to_string()}

MONTHLY SALES
-------------
{monthly_sales.to_string()}

SALES ANOMALIES
---------------
Number of unusual sales records: {len(sales_anomalies)}

"""

# ==========================================
# 10. CREATE GEMINI CLIENT
# ==========================================

client = genai.Client(api_key=api_key)

# ==========================================
# 11. CREATE AI PROMPT
# ==========================================

prompt = f"""
You are Data Detective AI, an intelligent business data analyst.

Analyze the following retail sales information:

{report_data}

Create a concise business investigation report.

Include:

1. Overall business performance
2. Top-performing region
3. Top-performing category
4. Top products
5. Monthly sales trend
6. Important anomaly findings
7. Three business insights
8. Three questions that should be investigated next

Use simple professional language.

Do not invent numbers that are not present in the data.
"""

# ==========================================
# 12. SEND DATA TO GEMINI
# ==========================================

for attempt in range(3):

    try:

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt
        )

        # ==========================================
        # 13. DISPLAY AI REPORT
        # ==========================================

        print("\n")
        print("=" * 60)
        print("       DATA DETECTIVE AI - BUSINESS REPORT")
        print("=" * 60)

        print(response.text)

        print("=" * 60)

        # ==========================================
        # 14. SAVE AI REPORT
        # ==========================================

        with open(
            "data/ai_business_report.txt",
            "w",
            encoding="utf-8"
        ) as file:

            file.write(response.text)

        print("\nAI report saved successfully!")

        break

    except Exception as e:

        print(f"\nAttempt {attempt + 1} failed.")

        if attempt < 2:

            print("Retrying in 10 seconds...")

            time.sleep(10)

        else:

            print("\nGemini is currently unavailable.")

            print(e)