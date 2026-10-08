# 🏦 Bank Marketing Prediction

##  Project Overview

Bank Marketing Prediction is a Machine Learning classification project that predicts whether a bank customer will subscribe to a term deposit or not.

The prediction is based on customer personal information, financial details, and marketing campaign information.

##  Project Objective

The main objective of this project is to identify whether a customer is likely to subscribe to a term deposit.

This can help banks:
- Identify potential customers
- Improve targeted marketing
- Save time and resources
- Make marketing campaigns more effective

##  Dataset Information

- Dataset: Bank Marketing Dataset
- Source: Kaggle
- Total Records: 45,211
- Input Features: 16
- Total Columns: 17
- Problem Type: Binary Classification
- Target Variable: `y`

### Target Variable

- `Yes` → Customer subscribed to a term deposit
- `No` → Customer did not subscribe to a term deposit

## Features Used

The main features used in the project are:

- Age
- Job
- Marital Status
- Education
- Credit Default
- Account Balance
- Housing Loan
- Personal Loan
- Contact
- Day
- Month
- Call Duration
- Campaign
- Previous Contact Days
- Previous Contacts
- Previous Campaign Outcome

##  Exploratory Data Analysis

Exploratory Data Analysis (EDA) was performed to understand the dataset and customer patterns.

The analysis included:

- Term Deposit Subscription Distribution
- Customer Distribution by Job
- Education Distribution
- Age Distribution
- Housing Loan Distribution
- Housing Loan vs Subscription Analysis

##  Data Preprocessing

The following preprocessing steps were performed:

- Checked the dataset structure
- Checked data types
- Handled categorical variables
- Encoded categorical features into numerical values
- Prepared the data for Machine Learning
- Split the dataset into training and testing data

##  Encoding

Categorical values such as job, education, marital status, housing loan, and other categorical features were converted into numerical form so that they could be processed by the Machine Learning model.

##  Machine Learning

This project uses Machine Learning classification techniques to predict whether a customer will subscribe to a term deposit.

Different classification models were trained and evaluated, and the best-performing model was selected for the final prediction system.

##  Prediction Workflow

Customer Details  
↓  
Data Preprocessing  
↓  
Encoding  
↓  
Machine Learning Model  
↓  
Prediction  
↓  
Yes / No

##  Streamlit Web Application

A user-friendly web application was developed using Streamlit.

Live Demo: https://bank-marketing-ml-nku2cyfa6daqmnrh3wapprc.streamlit.app/

The application contains the following sections:

-  Home
-  Dataset Insights
-  Customer Prediction
-  About Project

###  Customer Prediction

In the Customer Prediction section, the user can enter customer information such as:

- Age
- Job
- Marital Status
- Education
- Account Balance
- Housing Loan
- Personal Loan
- Contact Information
- Campaign Information

The application then predicts whether the customer is likely to subscribe to a term deposit.

##  Project Usage

This project is mainly useful for banks and bank marketing teams.

It helps to:

- Identify potential customers
- Perform targeted marketing
- Reduce unnecessary customer contacts
- Save time and marketing resources
- Improve marketing effectiveness

##  Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Jupyter Notebook

##  Project Structure

```text
Bank-Marketing-ML/
│
├── bank_app.py
├── bank_pipeline.pkl
├── bank-full.xlsx
├── requirements.txt
└── README.md
