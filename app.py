import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# NAVY BLUE + SKY BLUE + WHITE THEME
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #F8FAFC;
    color: #0F172A;
}

section[data-testid="stSidebar"] {
    background-color: #0B1F3A;
    border-right: 2px solid #38BDF8;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

h1 {
    color: #0B1F3A !important;
}

h2, h3 {
    color: #0369A1 !important;
}

p, label {
    color: #1E293B !important;
}

div[data-testid="metric-container"] {
    background-color: white;
    border: 1px solid #BAE6FD;
    border-radius: 12px;
    padding: 15px;
}

div[data-testid="metric-container"] label {
    color: #0369A1 !important;
}

div[data-testid="metric-container"] div {
    color: #0B1F3A !important;
}

.stButton > button {
    background-color: #0284C7;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #0369A1;
    color: white;
}

div[data-baseweb="select"] > div {
    background-color: white !important;
    border: 1px solid #BAE6FD !important;
}

div[data-testid="stNumberInput"] input {
    background-color: white !important;
    color: #0F172A !important;
    border: 1px solid #BAE6FD !important;
}

hr {
    border-color: #BAE6FD;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_excel("bank-full (1).xlsx")


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("bank_pipeline.pkl")


# ============================================================
# LOAD DATA AND MODEL
# ============================================================

try:

    df = load_data()
    pipeline = load_model()

except Exception as e:

    st.error("Unable to load dataset or model.")

    st.code(str(e))

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🏦 Bank Marketing")

st.sidebar.write(
    "Bank Marketing Prediction System"
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Dataset Insights",
        "Customer Prediction",
        "About Project"
    ]
)


# ============================================================
# HOME
# ============================================================

if page == "Home":

    st.title("🏦 Bank Marketing Prediction")

    st.write(
        "Predict whether a customer will subscribe to a term deposit "
        "using Machine Learning."
    )

    st.markdown("---")

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Customers",
            f"{len(df):,}"
        )

    with col2:

        st.metric(
            "Features",
            df.shape[1] - 1
        )

    with col3:

        subscribed = (df["y"] == "yes").sum()

        st.metric(
            "Subscribed",
            f"{subscribed:,}"
        )

    with col4:

        not_subscribed = (df["y"] == "no").sum()

        st.metric(
            "Not Subscribed",
            f"{not_subscribed:,}"
        )

    st.markdown("---")

    # --------------------------------------------------------
    # PROJECT OBJECTIVE
    # --------------------------------------------------------

    st.subheader("Project Objective")

    st.write(
        """
        The objective of this project is to predict whether a bank customer
        will subscribe to a term deposit based on customer information,
        financial details and marketing campaign information.
        """
    )

    # --------------------------------------------------------
    # TARGET VARIABLE
    # --------------------------------------------------------

    st.subheader("Target Variable")

    st.info(
        "y → Term Deposit Subscription (Yes / No)"
    )

    # --------------------------------------------------------
    # DATASET INFORMATION
    # --------------------------------------------------------

    st.subheader("Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.write(
            f"**Rows:** {df.shape[0]:,}"
        )

    with col2:

        st.write(
            f"**Columns:** {df.shape[1]}"
        )

    with col3:

        st.write(
            "**Problem Type:** Binary Classification"
        )


# ============================================================
# DATASET INSIGHTS
# ============================================================

elif page == "Dataset Insights":

    st.title("Dataset Insights")

    st.write(
        "Explore the Bank Marketing dataset."
    )

    st.markdown("---")

    # --------------------------------------------------------
    # DATASET PREVIEW
    # --------------------------------------------------------

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.markdown("---")

    # --------------------------------------------------------
    # TARGET DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("Term Deposit Subscription")

    target_counts = df["y"].value_counts()

    col1, col2 = st.columns(2)

    with col1:

        st.write("Subscription Count")

        st.bar_chart(
            target_counts
        )

    with col2:

        fig, ax = plt.subplots()

        ax.pie(
            target_counts.values,
            labels=target_counts.index,
            autopct="%1.1f%%"
        )

        ax.set_title(
            "Subscription Distribution"
        )

        st.pyplot(fig)

    st.markdown("---")

    # --------------------------------------------------------
    # JOB DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("Customers by Job")

    job_counts = df["job"].value_counts()

    st.bar_chart(
        job_counts
    )

    st.markdown("---")

    # --------------------------------------------------------
    # EDUCATION DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("Education Distribution")

    education_counts = df["education"].value_counts()

    st.bar_chart(
        education_counts
    )

    st.markdown("---")

    # --------------------------------------------------------
    # AGE DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("Age Distribution")

    fig, ax = plt.subplots()

    ax.hist(
        df["age"],
        bins=20
    )

    ax.set_xlabel("Age")

    ax.set_ylabel(
        "Number of Customers"
    )

    ax.set_title(
        "Customer Age Distribution"
    )

    st.pyplot(fig)

    st.markdown("---")

    # --------------------------------------------------------
    # HOUSING LOAN
    # --------------------------------------------------------

    st.subheader(
        "Housing Loan vs Subscription"
    )

    housing_table = pd.crosstab(
        df["housing"],
        df["y"]
    )

    st.dataframe(
        housing_table,
        use_container_width=True
    )

    st.bar_chart(
        housing_table
    )

    st.markdown("---")

    # --------------------------------------------------------
    # CONTACT TYPE
    # --------------------------------------------------------

    st.subheader("Contact Type")

    contact_counts = df["contact"].value_counts()

    st.bar_chart(
        contact_counts
    )

    st.markdown("---")

    # --------------------------------------------------------
    # CAMPAIGN BY MONTH
    # --------------------------------------------------------

    st.subheader("Campaign by Month")

    month_counts = df["month"].value_counts()

    st.bar_chart(
        month_counts
    )

    st.markdown("---")

    # --------------------------------------------------------
    # DATASET SUMMARY
    # --------------------------------------------------------

    st.subheader("Dataset Summary")

    st.dataframe(
        df.describe(
            include="all"
        ).T,
        use_container_width=True
    )


# ============================================================
# CUSTOMER PREDICTION
# ============================================================

elif page == "Customer Prediction":

    st.title("Customer Prediction")

    st.write(
        "Enter customer details and click Predict."
    )

    st.markdown("---")

    # --------------------------------------------------------
    # PERSONAL INFORMATION
    # --------------------------------------------------------

    st.subheader("Personal Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30
        )

    with col2:

        marital = st.selectbox(
            "Marital Status",
            [
                "married",
                "single",
                "divorced"
            ]
        )

    with col3:

        education = st.selectbox(
            "Education",
            [
                "primary",
                "secondary",
                "tertiary",
                "unknown"
            ]
        )

    # --------------------------------------------------------
    # EMPLOYMENT
    # --------------------------------------------------------

    st.subheader("Employment Information")

    col1, col2 = st.columns(2)

    with col1:

        job = st.selectbox(
            "Job",
            [
                "admin.",
                "blue-collar",
                "entrepreneur",
                "housemaid",
                "management",
                "retired",
                "self-employed",
                "services",
                "student",
                "technician",
                "unemployed",
                "unknown"
            ]
        )

    with col2:

        balance = st.number_input(
            "Account Balance",
            value=1000
        )

    # --------------------------------------------------------
    # LOAN INFORMATION
    # --------------------------------------------------------

    st.subheader("Loan Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        default = st.selectbox(
            "Credit Default",
            [
                "no",
                "yes"
            ]
        )

    with col2:

        housing = st.selectbox(
            "Housing Loan",
            [
                "yes",
                "no"
            ]
        )

    with col3:

        loan = st.selectbox(
            "Personal Loan",
            [
                "yes",
                "no"
            ]
        )

    # --------------------------------------------------------
    # CAMPAIGN INFORMATION
    # --------------------------------------------------------

    st.subheader("Campaign Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        contact = st.selectbox(
            "Contact",
            [
                "cellular",
                "telephone",
                "unknown"
            ]
        )

    with col2:

        day = st.number_input(
            "Contact Day",
            min_value=1,
            max_value=31,
            value=15
        )

    with col3:

        month = st.selectbox(
            "Month",
            [
                "jan",
                "feb",
                "mar",
                "apr",
                "may",
                "jun",
                "jul",
                "aug",
                "sep",
                "oct",
                "nov",
                "dec"
            ]
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        duration = st.number_input(
            "Call Duration",
            min_value=0,
            value=200
        )

    with col2:

        campaign = st.number_input(
            "Campaign Contacts",
            min_value=1,
            value=1
        )

    with col3:

        pdays = st.number_input(
            "Previous Contact Days",
            value=999
        )

    previous = st.number_input(
        "Previous Contacts",
        min_value=0,
        value=0
    )

    poutcome = st.selectbox(
        "Previous Campaign Outcome",
        [
            "failure",
            "other",
            "success",
            "unknown"
        ]
    )

    st.markdown("---")

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if st.button(
        "Predict",
        use_container_width=True
    ):

        input_data = pd.DataFrame({

            "age": [age],

            "job": [job],

            "marital": [marital],

            "education": [education],

            "default": [default],

            "balance": [balance],

            "housing": [housing],

            "loan": [loan],

            "contact": [contact],

            "day": [day],

            "month": [month],

            "duration": [duration],

            "campaign": [campaign],

            "pdays": [pdays],

            "previous": [previous],

            "poutcome": [poutcome]

        })

        try:

            prediction = pipeline.predict(
                input_data
            )

            st.markdown("---")

            st.subheader("Prediction Result")

            if prediction[0] == "yes":

                st.success(
                    "Customer is likely to subscribe to the Term Deposit."
                )

            else:

                st.error(
                    "Customer is unlikely to subscribe to the Term Deposit."
                )

            st.markdown("---")

            st.subheader("Customer Details")

            st.dataframe(
                input_data.T.rename(
                    columns={
                        0: "Value"
                    }
                ),
                use_container_width=True
            )

        except Exception as e:

            st.error(
                "Prediction failed."
            )

            st.code(
                str(e)
            )


# ============================================================
# ABOUT PROJECT
# ============================================================

elif page == "About Project":

    st.title(
        "About Bank Marketing Prediction"
    )

    st.subheader(
        "Project Description"
    )

    st.write(
        """
        This project uses Machine Learning to predict whether a bank
        customer will subscribe to a term deposit after a marketing campaign.
        """
    )

    st.subheader(
        "Objective"
    )

    st.write(
        """
        The main objective is to identify customers who are more likely
        to subscribe to a term deposit and help banks improve their
        marketing campaign efficiency.
        """
    )

    st.subheader(
        "Dataset"
    )

    st.write(
        """
        The dataset contains customer demographic information,
        financial details, communication information and previous
        marketing campaign results.
        """
    )

    st.write(
        f"Dataset Size: {df.shape[0]:,} rows × {df.shape[1]} columns"
    )

    st.subheader(
        "Machine Learning"
    )

    st.write(
        """
        This is a Binary Classification problem.

        Target variable:

        Yes → Customer subscribed

        No → Customer did not subscribe
        """
    )

    st.subheader(
        "Important Features"
    )

    st.write(
        """
        Age
        Job
        Education
        Account Balance
        Housing Loan
        Personal Loan
        Contact Type
        Call Duration
        Campaign Contacts
        Previous Campaign Outcome
        """
    )

    st.subheader(
        "Business Use"
    )

    st.write(
        """
        Banks can use this prediction system to identify potential
        customers, reduce unnecessary marketing efforts and improve
        campaign effectiveness.
        """
    )

    st.success(
        "Bank Marketing Prediction - Machine Learning Project"
    )