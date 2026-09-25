import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned data
df = pd.read_csv("data/cleaned_sales.csv")

# Convert date column
df["Order_Date"] = pd.to_datetime(
    df["Order_Date"],
    dayfirst=True
)


# Create dashboard
fig, axes = plt.subplots(3, 2, figsize=(18, 14))

# -------------------------------------------------
# 1. Sales by Region
# -------------------------------------------------

region_sales = df.groupby("Region")["Sales"].sum()

region_sales.plot(
    kind="bar",
    ax=axes[0, 0]
)

axes[0, 0].set_title("Sales by Region")
axes[0, 0].set_xlabel("Region")
axes[0, 0].set_ylabel("Total Sales")


# -------------------------------------------------
# 2. Sales by Category
# -------------------------------------------------

category_sales = df.groupby("Category")["Sales"].sum()

category_sales.plot(
    kind="bar",
    ax=axes[0, 1]
)

axes[0, 1].set_title("Sales by Category")
axes[0, 1].set_xlabel("Category")
axes[0, 1].set_ylabel("Total Sales")


# -------------------------------------------------
# 3. Top 10 Products by Sales
# -------------------------------------------------

top_products = (
    df.groupby("Product")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

top_products.plot(
    kind="bar",
    ax=axes[1, 0]
)

axes[1, 0].set_title("Top 10 Products by Sales")
axes[1, 0].set_xlabel("Product")
axes[1, 0].set_ylabel("Total Sales")

axes[1, 0].tick_params(
    axis="x",
    rotation=45
)


# -------------------------------------------------
# 4. Sales Distribution
# -------------------------------------------------

axes[1, 1].hist(
    df["Sales"],
    bins=20
)

axes[1, 1].set_title("Sales Distribution")
axes[1, 1].set_xlabel("Sales")
axes[1, 1].set_ylabel("Frequency")


# -------------------------------------------------
# 5. Discount vs Sales
# -------------------------------------------------

axes[2, 0].scatter(
    df["Discount"],
    df["Sales"]
)

axes[2, 0].set_title("Discount vs Sales")
axes[2, 0].set_xlabel("Discount")
axes[2, 0].set_ylabel("Sales")


# -------------------------------------------------
# 6. Monthly Sales Trend
# -------------------------------------------------

monthly_sales = (
    df.groupby(
        df["Order_Date"].dt.to_period("M")
    )["Sales"]
    .sum()
)

monthly_sales.index = monthly_sales.index.astype(str)

monthly_sales.plot(
    kind="line",
    marker="o",
    ax=axes[2, 1]
)

axes[2, 1].set_title("Monthly Sales Trend")
axes[2, 1].set_xlabel("Month")
axes[2, 1].set_ylabel("Total Sales")

axes[2, 1].tick_params(
    axis="x",
    rotation=45
)


# -------------------------------------------------
# Dashboard Title
# -------------------------------------------------

fig.suptitle(
    "Retail Sales Analysis Dashboard",
    fontsize=22
)


# Adjust layout
plt.tight_layout(
    rect=[0, 0, 1, 0.96]
)

# Show dashboard
plt.show()