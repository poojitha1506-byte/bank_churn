Bank Customer Churn Analysis: A Statistical and Machine Learning Approach*

## Project Overview

This project focuses on analyzing bank customer data to understand the factors influencing customer churn (customer attrition) and to build predictive models for identifying customers who are likely to leave the bank.

The study combines statistical techniues and machine learning models to extract insights and support data-driven decision-making in the banking sector.

## Objectives

* To preprocess the dataset (handling missing values, outliers, encoding, scaling)
* To perform Exploratory Data Analysis (EDA) and visualize patterns
* To apply statistical tests (Chi-square test, ANOVA, t-test)
* To perform customer segmentation using K-Means clustering
* To build and compare machine learning models for churn prediction
* To provide insights and recommendations for customer retention


## Dataset

* **Source:** Kaggle (Customer Churn Dataset by bhuviranga) 
* **Rows:** 10,000
* **Columns:** 12

### Features include:

* Demographics: Age, Gender, Country
* Account Info: Balance, Tenure, Number of Products
* Financial Info: Credit Score, Estimated Salary
* Activity: Credit Card, Active Member
* **Target Variable:** Churn (0 = No, 1 = Yes)

##  Technologies & Tools

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn
* Statsmodels

## Data Preprocessing

* Checked missing values and duplicates
* Outlier detection using IQR method and winsorization
* Feature selection and cleaning
* Encoding categorical variables
* Feature scaling

## Exploratory Data Analysis

* Distribution analysis using histograms and boxplots
* Churn distribution visualization
* Correlation heatmap
* Feature vs churn analysis (Age, Balance, Credit Score, etc.)

## Statistical Analysis

* **Chi-Square Test** → relationship between categorical variables and churn
* **ANOVA** → compare means across groups
* **t-test** → numerical feature comparison

## Clustering

* **K-Means Clustering**

  * Segmented customers into groups
  * Analyzed churn behavior across clusters
  * 
## Machine Learning Models

* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest
* Gradient Boosting

## Model Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

## Key Insights

* Customer activity and engagement strongly influence churn
* Balance and number of products impact retention
* Certain customer segments show higher churn rates
* Machine learning models effectively predict churn behavior
  

## 📌 Future Scope

* Apply advanced models like XGBoost
* Improve model performance using hyperparameter tuning
* Deploy model using Streamlit or Flask
* Use real-time banking datasets

