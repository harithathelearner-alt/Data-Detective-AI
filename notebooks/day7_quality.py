import pandas as pd

# Load cleaned data
df = pd.read_csv("data/cleaned_sales.csv")

print("========== DATA QUALITY REPORT ==========")

# 1. Dataset size
print("\n1. Dataset Size")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# 2. Missing values
print("\n2. Missing Values")
print(df.isnull().sum())

# 3. Duplicate rows
print("\n3. Duplicate Rows")
print(df.duplicated().sum())

# 4. Data types
print("\n4. Data Types")
print(df.dtypes)

# 5. Negative values
print("\n5. Negative Values")

numeric_columns = [
    "Quantity",
    "Unit_Price",
    "Discount",
    "Sales"
]

print(df[numeric_columns].lt(0).sum())

# 6. Zero values
print("\n6. Zero Values")
print((df[numeric_columns] == 0).sum())

print("\n========== QUALITY CHECK COMPLETED ==========")
# 7. Sales Anomaly Detection using IQR

Q1 = df["Sales"].quantile(0.25)
Q3 = df["Sales"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

anomalies = df[
    (df["Sales"] < lower_limit) |
    (df["Sales"] > upper_limit)
]

print("\n7. Sales Anomalies")
print("Lower Limit:", lower_limit)
print("Upper Limit:", upper_limit)
print("Number of Anomalies:", len(anomalies))

print("\nAnomalous Orders:")
print(anomalies[[
    "Order_ID",
    "Order_Date",
    "Region",
    "Category",
    "Product",
    "Sales"
]])
# 8. Quantity Anomaly Detection

Q1_qty = df["Quantity"].quantile(0.25)
Q3_qty = df["Quantity"].quantile(0.75)

IQR_qty = Q3_qty - Q1_qty

lower_qty = Q1_qty - 1.5 * IQR_qty
upper_qty = Q3_qty + 1.5 * IQR_qty

quantity_anomalies = df[
    (df["Quantity"] < lower_qty) |
    (df["Quantity"] > upper_qty)
]

print("\n8. Quantity Anomalies")
print("Lower Limit:", lower_qty)
print("Upper Limit:", upper_qty)
print("Number of Quantity Anomalies:", len(quantity_anomalies))

print("\nQuantity Anomalous Orders:")
print(quantity_anomalies[
    ["Order_ID", "Product", "Quantity", "Sales"]
])
# 9. Discount Anomaly Detection

Q1_discount = df["Discount"].quantile(0.25)
Q3_discount = df["Discount"].quantile(0.75)

IQR_discount = Q3_discount - Q1_discount

lower_discount = Q1_discount - 1.5 * IQR_discount
upper_discount = Q3_discount + 1.5 * IQR_discount

discount_anomalies = df[
    (df["Discount"] < lower_discount) |
    (df["Discount"] > upper_discount)
]

print("\n9. Discount Anomalies")
print("Lower Limit:", lower_discount)
print("Upper Limit:", upper_discount)
print("Number of Discount Anomalies:", len(discount_anomalies))

print("\nDiscount Anomalous Orders:")
print(discount_anomalies[
    ["Order_ID", "Product", "Discount", "Sales"]
])
# 10. Unit Price Anomaly Detection

Q1_price = df["Unit_Price"].quantile(0.25)
Q3_price = df["Unit_Price"].quantile(0.75)

IQR_price = Q3_price - Q1_price

lower_price = Q1_price - 1.5 * IQR_price
upper_price = Q3_price + 1.5 * IQR_price

price_anomalies = df[
    (df["Unit_Price"] < lower_price) |
    (df["Unit_Price"] > upper_price)
]

print("\n10. Unit Price Anomalies")
print("Lower Limit:", lower_price)
print("Upper Limit:", upper_price)
print("Number of Price Anomalies:", len(price_anomalies))

print("\nPrice Anomalous Orders:")
print(price_anomalies[
    ["Order_ID", "Product", "Unit_Price", "Sales"]
])
print("\n========== ANOMALY SUMMARY ==========")

print("Sales anomalies:", len(anomalies))
print("Quantity anomalies:", len(quantity_anomalies))
print("Discount anomalies:", len(discount_anomalies))
print("Unit Price anomalies:", len(price_anomalies))

print("\n========== DATA DETECTIVE CHECK COMPLETED ==========")
# Save anomaly results

anomalies.to_csv(
    "data/sales_anomalies.csv",
    index=False
)

quantity_anomalies.to_csv(
    "data/quantity_anomalies.csv",
    index=False
)

discount_anomalies.to_csv(
    "data/discount_anomalies.csv",
    index=False
)

price_anomalies.to_csv(
    "data/price_anomalies.csv",
    index=False
)

print("\nAnomaly files saved successfully!")
print("\n========== DATA DETECTIVE SUMMARY ==========")

print(f"""
Dataset contains {len(df)} rows and {len(df.columns)} columns.

Missing values: {df.isnull().sum().sum()}
Duplicate rows: {df.duplicated().sum()}

Sales anomalies: {len(anomalies)}
Quantity anomalies: {len(quantity_anomalies)}
Discount anomalies: {len(discount_anomalies)}
Unit Price anomalies: {len(price_anomalies)}
""")

print("Data Detective analysis completed successfully!")
# Create anomaly summary table

anomaly_summary = pd.DataFrame({
    "Anomaly_Type": [
        "Sales",
        "Quantity",
        "Discount",
        "Unit_Price"
    ],
    "Number_of_Anomalies": [
        len(anomalies),
        len(quantity_anomalies),
        len(discount_anomalies),
        len(price_anomalies)
    ]
})

anomaly_summary.to_csv(
    "data/anomaly_summary.csv",
    index=False
)

print("\nAnomaly summary saved to data/anomaly_summary.csv")
print(anomaly_summary)
# Final anomaly finding

total_anomalies = (
    len(anomalies)
    + len(quantity_anomalies)
    + len(discount_anomalies)
    + len(price_anomalies)
)

print("\n========== FINAL FINDING ==========")

print("Total anomaly records detected:", total_anomalies)

if total_anomalies == 0:
    print("No significant anomalies detected.")
else:
    print("Potential unusual records were detected.")
    print("These records should be investigated further.")

print("\nData Detective AI - Investigation Complete!")