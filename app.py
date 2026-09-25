import os
import io
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Data Detective AI",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# GEMINI API SETUP
# =========================================================

load_dotenv(dotenv_path=".env")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("Gemini API key not found in .env file.")
    st.stop()

client = genai.Client(api_key=api_key)


# =========================================================
# SESSION STATE
# =========================================================

if "ai_report" not in st.session_state:
    st.session_state.ai_report = ""

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False


# =========================================================
# TITLE
# =========================================================

st.title(" Data Detective AI")

st.markdown(
    """
    **AI-Powered Data Analysis & Business Investigation**

    Upload your dataset, investigate data quality,
    discover patterns and anomalies, visualize trends,
    and ask questions using natural language.
    """
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header(" Data Detective AI")

    st.write("### Features")

    st.write(" Data Upload")
    st.write(" Data Quality")
    st.write(" Data Analysis")
    st.write(" Visualization")
    st.write(" Anomaly Detection")
    st.write(" AI Investigation")
    st.write(" AI Data Q&A")
    st.write(" Report Download")

    st.divider()

    st.caption(
        "Built with Python, Pandas, Streamlit & Gemini AI"
    )


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    " Upload your CSV or Excel file",
    type=["csv", "xlsx"]
)


# =========================================================
# IF FILE UPLOADED
# =========================================================

if uploaded_file is not None:

    # -----------------------------------------------------
    # READ FILE
    # -----------------------------------------------------

    try:

        if uploaded_file.name.lower().endswith(".csv"):

            df = pd.read_csv(uploaded_file)

        else:

            df = pd.read_excel(uploaded_file)

    except Exception as e:

        st.error(
            f"Unable to read the uploaded file: {e}"
        )

        st.stop()


    st.success(
        f"Dataset uploaded successfully!  "
        f"({uploaded_file.name})"
    )


    # =====================================================
    # DATASET OVERVIEW
    # =====================================================

    st.header(" Dataset Overview")

    total_rows = df.shape[0]
    total_columns = df.shape[1]
    total_missing = int(
        df.isnull().sum().sum()
    )
    total_duplicates = int(
        df.duplicated().sum()
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Rows",
            f"{total_rows:,}"
        )

    with col2:
        st.metric(
            "Columns",
            f"{total_columns:,}"
        )

    with col3:
        st.metric(
            "Missing Values",
            f"{total_missing:,}"
        )

    with col4:
        st.metric(
            "Duplicate Rows",
            f"{total_duplicates:,}"
        )


    # =====================================================
    # DATA PREVIEW
    # =====================================================

    st.header(" Data Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


    # =====================================================
    # COLUMN INFORMATION
    # =====================================================

    st.header(" Column Information")

    column_info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [
            str(dtype)
            for dtype in df.dtypes
        ],
        "Missing Values": [
            int(df[column].isnull().sum())
            for column in df.columns
        ],
        "Unique Values": [
            int(df[column].nunique())
            for column in df.columns
        ]
    })

    st.dataframe(
        column_info,
        use_container_width=True
    )


    # =====================================================
    # DATA QUALITY
    # =====================================================

    st.header(" Data Quality Investigation")

    quality_col1, quality_col2 = st.columns(2)

    with quality_col1:

        st.subheader("Missing Values")

        missing_values = (
            df.isnull()
            .sum()
            .sort_values(ascending=False)
        )

        missing_display = (
            missing_values
            .to_frame("Missing Count")
        )

        st.dataframe(
            missing_display,
            use_container_width=True
        )


    with quality_col2:

        st.subheader("Duplicate Records")

        st.metric(
            "Duplicate Rows",
            total_duplicates
        )

        if total_duplicates > 0:

            st.warning(
                "Duplicate records detected."
            )

        else:

            st.success(
                "No duplicate records detected. "
            )


    # =====================================================
    # CLEAN DATA
    # =====================================================

    st.header(" Data Cleaning")

    clean_df = df.copy()

    cleaning_col1, cleaning_col2 = st.columns(2)

    with cleaning_col1:

        missing_before = int(
            clean_df.isnull().sum().sum()
        )

        if missing_before > 0:

            st.warning(
                f"Missing values found: "
                f"{missing_before:,}"
            )

            clean_df = clean_df.dropna()

        else:

            st.success(
                "No missing values found. "
            )


    with cleaning_col2:

        duplicates_before = int(
            clean_df.duplicated().sum()
        )

        if duplicates_before > 0:

            clean_df = clean_df.drop_duplicates()

            st.warning(
                f"Removed duplicate rows: "
                f"{duplicates_before:,}"
            )

        else:

            st.success(
                "No duplicates to remove. "
            )


    st.write(
        f"Cleaned dataset size: "
        f"**{clean_df.shape[0]:,} rows × "
        f"{clean_df.shape[1]:,} columns**"
    )


    # =====================================================
    # CLEANED DATA DOWNLOAD
    # =====================================================

    clean_csv = clean_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download Cleaned Dataset",
        data=clean_csv,
        file_name="cleaned_data_detective_dataset.csv",
        mime="text/csv"
    )


    # =====================================================
    # NUMERICAL SUMMARY
    # =====================================================

    st.header(" Statistical Summary")

    numeric_columns = (
        df.select_dtypes(
            include="number"
        ).columns.tolist()
    )

    if len(numeric_columns) > 0:

        st.dataframe(
            df[numeric_columns].describe(),
            use_container_width=True
        )

    else:

        st.info(
            "No numerical columns found."
        )


    # =====================================================
    # SALES / BUSINESS ANALYSIS
    # =====================================================

    if "Sales" in df.columns:

        st.header(" Business Performance")

        total_sales = df["Sales"].sum()

        average_sales = df["Sales"].mean()

        median_sales = df["Sales"].median()

        if "Quantity" in df.columns:

            total_quantity = df["Quantity"].sum()

        else:

            total_quantity = 0


        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Total Sales",
                f"{total_sales:,.2f}"
            )

        with col2:

            st.metric(
                "Average Sales",
                f"{average_sales:,.2f}"
            )

        with col3:

            st.metric(
                "Median Sales",
                f"{median_sales:,.2f}"
            )

        with col4:

            st.metric(
                "Total Quantity",
                f"{total_quantity:,.0f}"
            )


    # =====================================================
    # REGION ANALYSIS
    # =====================================================

    region_sales = None

    if (
        "Region" in df.columns
        and "Sales" in df.columns
    ):

        st.header(" Regional Sales Analysis")

        region_sales = (
            df.groupby("Region")["Sales"]
            .sum()
            .sort_values(ascending=False)
        )

        st.bar_chart(region_sales)

        top_region = region_sales.idxmax()
        top_region_sales = region_sales.max()

        st.info(
            f"Highest sales region: **{top_region}** "
            f"with **{top_region_sales:,.2f}** sales."
        )


    # =====================================================
    # CATEGORY ANALYSIS
    # =====================================================

    category_sales = None

    if (
        "Category" in df.columns
        and "Sales" in df.columns
    ):

        st.header(" Category Sales Analysis")

        category_sales = (
            df.groupby("Category")["Sales"]
            .sum()
            .sort_values(ascending=False)
        )

        st.bar_chart(category_sales)

        top_category = category_sales.idxmax()
        top_category_sales = category_sales.max()

        st.info(
            f"Highest sales category: **{top_category}** "
            f"with **{top_category_sales:,.2f}** sales."
        )


    # =====================================================
    # TOP PRODUCTS
    # =====================================================

    top_products = None

    if (
        "Product" in df.columns
        and "Sales" in df.columns
    ):

        st.header(" Top Products")

        top_products = (
            df.groupby("Product")["Sales"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )

        st.bar_chart(top_products)

        st.dataframe(
            top_products
            .rename("Total Sales")
            .to_frame(),
            use_container_width=True
        )


    # =====================================================
    # MONTHLY SALES TREND
    # =====================================================

    monthly_sales = None

    if (
        "Order_Date" in df.columns
        and "Sales" in df.columns
    ):

        st.header(" Monthly Sales Trend")

        temp_df = df.copy()

        temp_df["Order_Date"] = pd.to_datetime(
            temp_df["Order_Date"],
            dayfirst=True,
            errors="coerce"
        )

        valid_dates = temp_df[
            temp_df["Order_Date"].notna()
        ]

        if len(valid_dates) > 0:

            monthly_sales = (
                valid_dates
                .groupby(
                    valid_dates["Order_Date"]
                    .dt.to_period("M")
                )["Sales"]
                .sum()
            )

            monthly_sales.index = (
                monthly_sales.index.astype(str)
            )

            st.line_chart(
                monthly_sales
            )

            highest_month = (
                monthly_sales.idxmax()
            )

            highest_month_sales = (
                monthly_sales.max()
            )

            lowest_month = (
                monthly_sales.idxmin()
            )

            lowest_month_sales = (
                monthly_sales.min()
            )

            col1, col2 = st.columns(2)

            with col1:

                st.success(
                    f" Highest month: "
                    f"{highest_month} — "
                    f"{highest_month_sales:,.2f}"
                )

            with col2:

                st.warning(
                    f" Lowest month: "
                    f"{lowest_month} — "
                    f"{lowest_month_sales:,.2f}"
                )


    # =====================================================
    # CORRELATION ANALYSIS
    # =====================================================

    if len(numeric_columns) >= 2:

        st.header(" Correlation Analysis")

        correlation = (
            df[numeric_columns]
            .corr()
        )

        st.dataframe(
            correlation,
            use_container_width=True
        )


    # =====================================================
    # ANOMALY DETECTION
    # =====================================================

    st.header(" Anomaly Investigation")

    anomaly_results = {}

    anomaly_columns = [
        "Sales",
        "Quantity",
        "Discount",
        "Unit_Price"
    ]

    for column in anomaly_columns:

        if column in df.columns:

            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)

            IQR = Q3 - Q1

            lower_limit = (
                Q1 - 1.5 * IQR
            )

            upper_limit = (
                Q3 + 1.5 * IQR
            )

            anomalies = df[
                (
                    df[column] < lower_limit
                )
                |
                (
                    df[column] > upper_limit
                )
            ]

            anomaly_results[column] = {
                "data": anomalies,
                "Q1": Q1,
                "Q3": Q3,
                "IQR": IQR,
                "lower": lower_limit,
                "upper": upper_limit
            }


    # =====================================================
    # ANOMALY SUMMARY
    # =====================================================

    anomaly_summary = pd.DataFrame({
        "Anomaly Type": [
            column
            for column in anomaly_results
        ],
        "Number of Anomalies": [
            len(
                result["data"]
            )
            for result in anomaly_results.values()
        ]
    })


    if len(anomaly_summary) > 0:

        st.dataframe(
            anomaly_summary,
            use_container_width=True
        )


    total_anomalies = sum(
        len(result["data"])
        for result in anomaly_results.values()
    )


    if total_anomalies == 0:

        st.success(
            "No significant anomalies detected. "
        )

    else:

        st.warning(
            f"Potential unusual records detected: "
            f"{total_anomalies}"
        )


    # =====================================================
    # ANOMALY DETAILS
    # =====================================================

    for column, result in anomaly_results.items():

        anomalies = result["data"]

        if len(anomalies) > 0:

            with st.expander(
                f"🚨 {column} Anomalies "
                f"({len(anomalies)})"
            ):

                st.write(
                    f"Normal range: "
                    f"{result['lower']:,.2f} "
                    f"to "
                    f"{result['upper']:,.2f}"
                )

                st.dataframe(
                    anomalies,
                    use_container_width=True
                )


    # =====================================================
    # AI BUSINESS INVESTIGATION
    # =====================================================

    st.header(" AI Business Investigation")

    report_data = f"""
DATASET INFORMATION
-------------------
Rows: {df.shape[0]}
Columns: {df.shape[1]}

Columns:
{list(df.columns)}
"""


    # -----------------------------------------------------
    # BUSINESS METRICS
    # -----------------------------------------------------

    if "Sales" in df.columns:

        report_data += f"""

BUSINESS METRICS
----------------
Total Sales: {df['Sales'].sum():,.2f}
Average Sales: {df['Sales'].mean():,.2f}
Median Sales: {df['Sales'].median():,.2f}
"""


    if "Quantity" in df.columns:

        report_data += f"""
Total Quantity: {df['Quantity'].sum():,.0f}
"""


    # -----------------------------------------------------
    # REGION
    # -----------------------------------------------------

    if region_sales is not None:

        report_data += f"""

REGION SALES
------------
{region_sales.to_string()}
"""


    # -----------------------------------------------------
    # CATEGORY
    # -----------------------------------------------------

    if category_sales is not None:

        report_data += f"""

CATEGORY SALES
--------------
{category_sales.to_string()}
"""


    # -----------------------------------------------------
    # PRODUCTS
    # -----------------------------------------------------

    if top_products is not None:

        report_data += f"""

TOP PRODUCTS
------------
{top_products.to_string()}
"""


    # -----------------------------------------------------
    # MONTHLY
    # -----------------------------------------------------

    if monthly_sales is not None:

        report_data += f"""

MONTHLY SALES
-------------
{monthly_sales.to_string()}
"""


    # -----------------------------------------------------
    # ANOMALIES
    # -----------------------------------------------------

    report_data += """

ANOMALY SUMMARY
---------------
"""

    for column, result in anomaly_results.items():

        report_data += (
            f"{column}: "
            f"{len(result['data'])} anomalies\n"
        )


    # =====================================================
    # GENERATE AI REPORT
    # =====================================================

    if st.button(
        " Generate AI Business Investigation"
    ):

        prompt = f"""
You are Data Detective AI,
an intelligent business data analyst.

Analyze the following dataset summary.

{report_data}

Create a professional business investigation report.

Include:

1. Overall business performance
2. Important numerical findings
3. Top-performing region
4. Top-performing category
5. Top products
6. Monthly sales trend
7. Anomaly findings
8. Three data-driven business insights
9. Three questions that should be investigated next

Rules:

- Use ONLY the provided data.
- Do not invent numbers.
- Clearly distinguish facts from hypotheses.
- Do not claim fraud, causes, or business reasons
  unless the data actually proves them.
- Use simple professional language.
"""

        with st.spinner(
            " Data Detective AI is investigating..."
        ):

            try:

                response = client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=prompt
                )

                st.session_state.ai_report = (
                    response.text
                )

            except Exception as e:

                st.error(
                    f"AI report generation failed: {e}"
                )


    # =====================================================
    # DISPLAY AI REPORT
    # =====================================================

    if st.session_state.ai_report:

        st.subheader(
            " AI Investigation Report"
        )

        st.markdown(
            st.session_state.ai_report
        )

        report_download = (
            st.session_state.ai_report
            .encode("utf-8")
        )

        st.download_button(
            label="📥 Download AI Report",
            data=report_download,
            file_name="data_detective_ai_report.txt",
            mime="text/plain"
        )


    # =====================================================
    # ASK DATA DETECTIVE AI
    # =====================================================

    st.header("💬 Ask Data Detective AI")

    st.write(
        "Ask a question about your uploaded dataset."
    )

    user_question = st.text_input(
        "Example: Which region has the highest sales?"
    )


    # =====================================================
    # AI QUESTION ANSWERING
    # =====================================================

    if user_question:

        # -------------------------------------------------
        # BUILD DATA SUMMARY
        # -------------------------------------------------

        data_summary = f"""
DATASET INFORMATION
-------------------
Rows: {df.shape[0]}
Columns: {df.shape[1]}

Columns:
{list(df.columns)}

SUMMARY STATISTICS
------------------
{df.describe().to_string()}
"""


        # -------------------------------------------------
        # TOTAL SALES
        # -------------------------------------------------

        if "Sales" in df.columns:

            data_summary += f"""

TOTAL SALES
-----------
{df['Sales'].sum():,.2f}
"""


        # -------------------------------------------------
        # REGION
        # -------------------------------------------------

        if region_sales is not None:

            data_summary += f"""

SALES BY REGION
---------------
{region_sales.to_string()}
"""


        # -------------------------------------------------
        # CATEGORY
        # -------------------------------------------------

        if category_sales is not None:

            data_summary += f"""

SALES BY CATEGORY
-----------------
{category_sales.to_string()}
"""


        # -------------------------------------------------
        # PRODUCTS
        # -------------------------------------------------

        if top_products is not None:

            data_summary += f"""

TOP 10 PRODUCTS
---------------
{top_products.to_string()}
"""


        # -------------------------------------------------
        # MONTHLY
        # -------------------------------------------------

        if monthly_sales is not None:

            data_summary += f"""

MONTHLY SALES
-------------
{monthly_sales.to_string()}
"""


        # -------------------------------------------------
        # ANOMALIES
        # -------------------------------------------------

        data_summary += """

ANOMALIES
---------
"""

        for column, result in anomaly_results.items():

            data_summary += (
                f"{column}: "
                f"{len(result['data'])} anomalies\n"
            )


        # =================================================
        # AI PROMPT
        # =================================================

        prompt = f"""
You are Data Detective AI.

Answer the user's question using ONLY
the data summary below.

DATA SUMMARY:

{data_summary}

USER QUESTION:

{user_question}

Rules:

1. Give the direct answer first.
2. Use exact numbers from the data.
3. Do not invent information.
4. If the answer cannot be determined,
   clearly say so.
5. Explain the supporting evidence.
6. Clearly distinguish facts from hypotheses.
7. Keep the response simple and professional.
"""

        with st.spinner(
            " Data Detective AI is investigating..."
        ):

            try:

                response = client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=prompt
                )

                st.subheader(
                    " AI Investigation Result"
                )

                st.write(
                    response.text
                )

            except Exception as e:

                st.error(
                    f"AI request failed: {e}"
                )


        # =================================================
        # SUPPORTING EVIDENCE
        # =================================================

        st.subheader(
            " Supporting Data Evidence"
        )


        if region_sales is not None:

            st.write("###  Sales by Region")

            st.dataframe(
                region_sales
                .rename("Total Sales")
                .to_frame(),
                use_container_width=True
            )


        if category_sales is not None:

            st.write("###  Sales by Category")

            st.dataframe(
                category_sales
                .rename("Total Sales")
                .to_frame(),
                use_container_width=True
            )


        if top_products is not None:

            st.write("###  Top Products")

            st.dataframe(
                top_products
                .rename("Total Sales")
                .to_frame(),
                use_container_width=True
            )


        # =================================================
        # NATURAL LANGUAGE → DATA VISUALIZATION
        # =================================================

        st.subheader(
            " Automatic Data Visualization"
        )

        question_lower = (
            user_question.lower()
        )


        # -------------------------------------------------
        # MONTHLY SALES TREND
        # -------------------------------------------------

        if (
            "monthly" in question_lower
            and "sales" in question_lower
        ):

            if monthly_sales is not None:

                st.write(
                    "###  Monthly Sales Trend"
                )

                st.line_chart(
                    monthly_sales
                )


        # -------------------------------------------------
        # REGION SALES
        # -------------------------------------------------

        elif (
            "region" in question_lower
            and "sales" in question_lower
        ):

            if region_sales is not None:

                st.write(
                    "###  Sales by Region"
                )

                st.bar_chart(
                    region_sales
                )


        # -------------------------------------------------
        # CATEGORY SALES
        # -------------------------------------------------

        elif (
            "category" in question_lower
            and "sales" in question_lower
        ):

            if category_sales is not None:

                st.write(
                    "###  Sales by Category"
                )

                st.bar_chart(
                    category_sales
                )


        # -------------------------------------------------
        # TOP PRODUCTS
        # -------------------------------------------------

        elif (
            "product" in question_lower
            and (
                "top" in question_lower
                or "highest" in question_lower
                or "best" in question_lower
            )
        ):

            if top_products is not None:

                st.write(
                    "###  Top 10 Products"
                )

                st.bar_chart(
                    top_products
                )


        # -------------------------------------------------
        # ANOMALY QUESTION
        # -------------------------------------------------

        elif (
            "anomal" in question_lower
            or "unusual" in question_lower
            or "outlier" in question_lower
        ):

            st.write(
                "###  Anomaly Investigation"
            )

            st.metric(
                "Total Potential Anomaly Records",
                total_anomalies
            )

            st.dataframe(
                anomaly_summary,
                use_container_width=True
            )


        # -------------------------------------------------
        # NO CHART MATCH
        # -------------------------------------------------

        else:

            st.info(
                "No automatic visualization matched "
                "this question."
            )


else:

    # =====================================================
    # NO FILE
    # =====================================================

    st.info(
        " Upload a CSV or Excel file to start "
        "your investigation."
    )

    st.markdown(
        """
        ###  What Data Detective AI can do

        **Analyze**  
        Understand your dataset and business metrics.

        ** Clean**  
        Detect missing values and duplicates.

        ** Investigate**  
        Find unusual records using anomaly detection.

        ** Visualize**  
        Discover trends and patterns.

        ** Ask AI**  
        Ask questions about your data using natural language.

        ** Generate Reports**  
        Create an AI-powered business investigation report.
        """
    )