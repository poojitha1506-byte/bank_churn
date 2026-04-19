#!/usr/bin/env python
# coding: utf-8

# In[2]:


"""✔ Pandas → used to load, clean, and manipulate the dataset
✔ NumPy → used for numerical operations
✔ Matplotlib & Seaborn → used to create graphs (histograms, scatter plots, boxplots)
✔ Warnings → used to hide unnecessary warnings"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# stats for ANOVA & post-hoc
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd 
from scipy import stats

"""These are all the machine learning tools
– splitting data
– scaling
– model building
– evaluating results
– tuning models"""
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder # preprocessing & PCA
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, classification_report, ConfusionMatrixDisplay)
from scipy.stats import chi2_contingency, ttest_ind
"""✔ PCA = to reduce dimensions (multivariate analysis)
✔ KMeans = to find clusters (unsupervised learning)"""
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

#Saves the best ML model into a file.
import joblib

# Optional libs (install outside this cell with %pip if needed)
try:
    import xgboost as xgb
except:
    xgb = None
try:
    import shap
except:
    shap = None
    


# In[3]:



# 1) Load data (one place)
df = pd.read_csv(r"C:\Users\Elite X2 360\Desktop\Bank_Customer_Churn_Dataset.csv") #Reads CSV and Stores it in a DataFrame
df.head() # Shows first few rows


# In[4]:


# ========== 1) Drop identification column ==========
# place this immediately after reading the csv (or right after df_clean is created)
if 'customer_id' in df.columns:
    df.drop(columns=['customer_id'], inplace=True)


# In[5]:


#Descriptive stats: compute only for numeric features 
# Define numeric_cols explicitly (ensure it does NOT include products_number, credit_card, active_member, churn)
numeric_cols = ['credit_score', 'age', 'tenure', 'balance', 'estimated_salary']
# If numeric_cols don't exist or were defined earlier, ensure unwanted columns are removed
numeric_cols = [c for c in numeric_cols if c in df.columns]

print("Numeric features used for descriptive stats:", numeric_cols)
print(df[numeric_cols].describe().round(3))


# In[6]:


df.info() # Shows datatypes, missing values


# In[7]:


print("Shape: ",df.shape,"\n")
print("Missing values: ",df.isnull().sum(),"\n")
print("Duplicate values: ",df.duplicated().sum())


# In[8]:


# Function to identify binary columns
def is_binary(col):
    unique_vals = df[col].dropna().unique()
    return set(unique_vals).issubset({0, 1})

# List numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Columns to exclude (IDs, binary variables)
exclude_cols = []

for col in numeric_cols:
    if is_binary(col):
        exclude_cols.append(col)
    if "id" in col.lower():   # any ID-like column
        exclude_cols.append(col)

# Final numeric columns for outlier detection
numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

outlier_summary = {}

# Compute outliers using IQR method
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower) | (df[col] > upper)][col]
    outlier_summary[col] = len(outliers)

outlier_summary


# In[9]:




# Winsorize (cap) continuous numeric columns
for col in outlier_summary:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    df[col] = df[col].clip(lower, upper)
    
# Confirm outliers gone numerically
post_outlier_summary = {}
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)][col]
    post_outlier_summary[col] = len(outliers)
print("\nOutlier counts after capping (should be 0):")
print(post_outlier_summary)    

# Round discrete variables after clipping

df['products_number'] = df['products_number'].clip(lower, upper).round().astype(int)

df['products_number'].unique()


# In[10]:


# Create an in-memory copy of the cleaned dataset
df_clean = df.copy()


# In[11]:


# 4) EDA (short & effective)
# Univariate numeric histograms + boxplots
# Columns to exclude from histogram & boxplot
exclude_cols = ['products_number', 'estimated_salary']

# Filter numeric columns
plot_cols = [col for col in numeric_cols if col not in exclude_cols]

print("Plotting numeric columns:", plot_cols)

# Univariate numeric histograms + boxplots for remaining columns
for col in plot_cols:
    plt.figure(figsize=(10,3))
    
    # Histogram
    plt.subplot(1,2,1)
    sns.histplot(df_clean[col], bins=30, kde=False)
    plt.title(f"Histogram: {col}")
    
    # Boxplot
    plt.subplot(1,2,2)
    sns.boxplot(x=df_clean[col])
    plt.title(f"Boxplot: {col}")
    
    plt.tight_layout()
    plt.show()


# In[12]:


# Churn distribution
plt.figure(figsize=(4,3))
sns.countplot(x='churn', data=df_clean)
plt.title("Churn distribution")
plt.show()


# In[13]:


# Numeric vs churn boxplots
for col in numeric_cols:
    plt.figure(figsize=(5,3))
    sns.boxplot(x='churn', y=col, data=df_clean)
    plt.title(f"{col} by churn")
    plt.show()


# In[14]:


# Correlation heatmap (exclude products_number) ==========
# Ensure 'products_number' or 'products_number' like name not included
corr_cols = [c for c in numeric_cols if c != 'products_number']
plt.figure(figsize=(7,5))
sns.heatmap(df_clean[corr_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Numeric correlation")
plt.show()


# In[15]:


# Bar Chart: Age vs Credit Score
df_clean['age_group'] = pd.cut(df_clean['age'], bins=[18,30,40,50,60], labels=["18–30","31–40","41–50","51–60"])

plt.figure(figsize=(6,4))
sns.barplot(x='age_group', y='credit_score', data=df_clean)
plt.title("Age Group vs Credit Score")
plt.xlabel("Age Group")
plt.ylabel("Average Credit Score")
plt.tight_layout()
plt.show()


# In[16]:


plt.figure(figsize=(6,4))
sns.barplot(x='age_group', y='estimated_salary', data=df_clean)
plt.title("Age Group vs Estimated Salary")
plt.xlabel("Age Group")
plt.ylabel("Average Estimated Salary")
plt.tight_layout()
plt.show()


# In[17]:


# 1) AGE vs CHURN (boxplot)
if 'age' in df_clean.columns and 'churn' in df_clean.columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='churn', y='age', data=df_clean)
    plt.title("Age vs Churn")
    plt.xlabel("Churn")
    plt.ylabel("Age")
    plt.tight_layout()
    plt.show()

# -----------------------
# 2) CREDIT SCORE vs CHURN (boxplot)
# -----------------------
if 'credit_score' in df_clean.columns and 'churn' in df_clean.columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='churn', y='credit_score', data=df_clean)
    plt.title("Credit Score vs Churn")
    plt.xlabel("Churn")
    plt.ylabel("Credit Score")
    plt.tight_layout()
    plt.show()

# -----------------------
# 3) BALANCE vs CHURN (boxplot)
# -----------------------
if 'balance' in df_clean.columns and 'churn' in df_clean.columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='churn', y='balance', data=df_clean)
    plt.title("Balance vs Churn")
    plt.xlabel("Churn")
    plt.ylabel("Balance")
    plt.tight_layout()
    plt.show()

# -----------------------
# 4) PRODUCTS NUMBER vs CHURN (boxplot)
# -----------------------
if 'products_number' in df_clean.columns and 'churn' in df_clean.columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='churn', y='products_number', data=df_clean)
    plt.title("Products Number vs Churn")
    plt.xlabel("Churn")
    plt.ylabel("Products Number")
    plt.tight_layout()
    plt.show()

# -----------------------
# 5) COUNTRY vs CHURN (bar chart showing churn rate)
# -----------------------
if 'country' in df_clean.columns and 'churn' in df_clean.columns:
    plt.figure(figsize=(7,4))
    # churn rate per country
    sns.barplot(x='country', y='churn', data=df_clean, estimator=np.mean)
    plt.title("Country vs Churn Rate")
    plt.ylabel("Churn Rate (mean of churn)")
    plt.xlabel("Country")
    plt.ylim(0, df_clean['churn'].max() if df_clean['churn'].dtype != bool else 1)
    plt.tight_layout()
    plt.show()

# -----------------------
# 6) GENDER vs CHURN (bar chart showing churn rate)
# -----------------------
if 'gender' in df_clean.columns and 'churn' in df_clean.columns:
    plt.figure(figsize=(6,4))
    sns.barplot(x='gender', y='churn', data=df_clean, estimator=np.mean)
    plt.title("Gender vs Churn Rate")
    plt.ylabel("Churn Rate")
    plt.xlabel("Gender")
    plt.tight_layout()
    plt.show()

# -----------------------
# 7) ACTIVE_MEMBER vs CHURN (bar chart showing churn rate)
# -----------------------
if 'active_member' in df_clean.columns and 'churn' in df_clean.columns:
    plt.figure(figsize=(6,4))
    sns.barplot(x='active_member', y='churn', data=df_clean, estimator=np.mean)
    plt.title("Active Member vs Churn Rate")
    plt.ylabel("Churn Rate")
    plt.xlabel("Active Member")
    plt.tight_layout()
    plt.show()

# -----------------------
# 8) Basic count bar charts for categorical variables (counts)
#    country, gender, churn, credit_card
# -----------------------
# Country count
if 'country' in df_clean.columns:
    plt.figure(figsize=(7,3))
    sns.countplot(x='country', data=df_clean)
    plt.title("Country Count")
    plt.xlabel("Country")
    plt.tight_layout()
    plt.show()

# Gender count
if 'gender' in df_clean.columns:
    plt.figure(figsize=(5,3))
    sns.countplot(x='gender', data=df_clean)
    plt.title("Gender Count")
    plt.xlabel("Gender")
    plt.tight_layout()
    plt.show()

# Churn count
if 'churn' in df_clean.columns:
    plt.figure(figsize=(4,3))
    sns.countplot(x='churn', data=df_clean)
    plt.title("Churn Count")
    plt.xlabel("Churn")
    plt.tight_layout()
    plt.show()

# Credit Card count
if 'credit_card' in df_clean.columns:
    plt.figure(figsize=(5,3))
    sns.countplot(x='credit_card', data=df_clean)
    plt.title("Credit Card Count")
    plt.xlabel("Credit Card")
    plt.tight_layout()
    plt.show()


# In[18]:


# 5) Statistical tests (chi-square for categories, t-test for numbers)
# ---------- CLEAN CHI-SQUARE TEST OUTPUT ----------
from scipy.stats import chi2_contingency

chi_square_cols = ['country', 'gender', 'credit_card', 'active_member', 'products_number']

print("---------- Chi-Square Test Results ----------\n")

for col in chi_square_cols:
    if col in df_clean.columns:
        table = pd.crosstab(df_clean[col], df_clean['churn'])
        chi2, p, dof, expected = chi2_contingency(table)
        
        print(f"Chi-square Test: {col} vs Churn")
        print(f"Chi-square: {chi2:.3f}   p-value: {p:.5f}\n")


# In[20]:


# ---- K-Means Clustering (Full Code) ----

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Select numeric features for clustering
features = ['age', 'balance', 'tenure', 'products_number', 'credit_score']
X = df_clean[features]

# 2. Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df_clean['kcluster'] = kmeans.fit_predict(X_scaled)

# 4. Churn rates for each cluster
print("\nChurn rates by cluster:")
display(df_clean.groupby('kcluster')['churn'].agg(['count', 'mean']).rename(columns={'mean':'churn_rate'}))

# 5. Cluster profiles (mean values of features)
print("\nCluster profiles:")
display(df_clean.groupby('kcluster')[features].mean())



# In[21]:


from scipy.stats import f_oneway

anova_cols = ['credit_score', 'age', 'tenure', 'balance']

for col in anova_cols:
    groups = [
        df_clean[df_clean['country'] == country][col].dropna()
        for country in df_clean['country'].unique()
    ]
    
    stat, p = f_oneway(*groups)
    print(f"ANOVA for {col} by country → F = {stat:.4f}, p = {p:.5f}")


# In[34]:


# 1. Prepare X, y
X = df_clean.drop(columns=['customer_id','churn'], errors='ignore')
y = df_clean['churn'].astype(int)

# 2. Detect numeric & categorical features (exclude binary numeric from "numeric" transformer)
def is_binary_series(s):
    vals = set(s.dropna().unique())
    return vals.issubset({0,1}) or vals.issubset({'0','1'})

num_features = [c for c in X.select_dtypes(include=['int64','float64']).columns if not is_binary_series(X[c])]
cat_features = [c for c in X.columns if c not in num_features]

print("Numeric features:", num_features)
print("Categorical features:", cat_features)

# 3. Preprocessor
numeric_transformer = Pipeline([('scaler', StandardScaler())])
categorical_transformer = Pipeline([('onehot', OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_features),
    ('cat', categorical_transformer, cat_features)
])

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)


# In[35]:


# --- 5) LOGISTIC REGRESSION (uses predefined X_train/X_test) ---
lr_pipe = Pipeline([
    ('pre', preprocessor),
    ('clf', LogisticRegression(max_iter=1000, random_state=RND))
])

lr_pipe.fit(X_train, y_train)
y_pred_lr = lr_pipe.predict(X_test)

cm_lr = confusion_matrix(y_test, y_pred_lr)
print("\nLogistic Regression - Confusion Matrix:\n", cm_lr)
print("\nClassification Report (Logistic Regression):")
print(classification_report(y_test, y_pred_lr, digits=4))

disp = ConfusionMatrixDisplay(confusion_matrix=cm_lr, display_labels=['Not Churn', 'Churn'])
fig, ax = plt.subplots(figsize=(5,4))
disp.plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title("Confusion Matrix — Logistic Regression")
plt.show()

# --- 6) SVM (uses same X_train/X_test) ---
svm_pipe = Pipeline([
    ('pre', preprocessor),
    ('clf', SVC(probability=True, random_state=RND))
])

svm_pipe.fit(X_train, y_train)
y_pred_svm = svm_pipe.predict(X_test)

cm_svm = confusion_matrix(y_test, y_pred_svm)
print("\nSVM - Confusion Matrix:\n", cm_svm)
print("\nClassification Report (SVM):")
print(classification_report(y_test, y_pred_svm, digits=4))

disp = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=['Not Churn', 'Churn'])
fig, ax = plt.subplots(figsize=(5,4))
disp.plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title("Confusion Matrix — SVM")
plt.show()

# --- 7) Random Forest (uses same X_train/X_test) ---
rf_pipe = Pipeline([
    ('pre', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=200, random_state=RND))
])

rf_pipe.fit(X_train, y_train)
y_pred_rf = rf_pipe.predict(X_test)

cm_rf = confusion_matrix(y_test, y_pred_rf)
print("\nRandom Forest - Confusion Matrix:\n", cm_rf)
print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf, digits=4))

disp = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=['Not Churn', 'Churn'])
fig, ax = plt.subplots(figsize=(5,4))
disp.plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title("Confusion Matrix — Random Forest")
plt.show()

# --- 8) Gradient Boosting (uses same X_train/X_test) ---
gb_pipe = Pipeline([
    ('pre', preprocessor),
    ('clf', GradientBoostingClassifier(n_estimators=200, random_state=RND))
])

gb_pipe.fit(X_train, y_train)
y_pred_gb = gb_pipe.predict(X_test)

cm_gb = confusion_matrix(y_test, y_pred_gb)
print("\nGradient Boosting - Confusion Matrix:\n", cm_gb)
print("\nClassification Report (Gradient Boosting):")
print(classification_report(y_test, y_pred_gb, digits=4))

disp = ConfusionMatrixDisplay(confusion_matrix=cm_gb, display_labels=['Not Churn', 'Churn'])
fig, ax = plt.subplots(figsize=(5,4))
disp.plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title("Confusion Matrix — Gradient Boosting")
plt.show()


# In[36]:


# === Evaluate fitted pipelines and pick best model(s) by metrics ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve)

# Collect the pipelines you saved / created earlier
model_pipes = {
    'LogisticRegression': globals().get('lr_pipe', None),
    'SVM': globals().get('svm_pipe', None),
    'RandomForest': globals().get('rf_pipe', None),
    'GradientBoosting': globals().get('gb_pipe', None)
}

# If any pipeline variable is missing, but a joblib file exists, try load it
import joblib, os
for name, pipe in list(model_pipes.items()):
    if pipe is None:
        fname = f"{name.lower().replace(' ', '_')}_pipe.joblib"
        if os.path.exists(fname):
            try:
                model_pipes[name] = joblib.load(fname)
                print(f"Loaded {name} from {fname}")
            except Exception as e:
                print(f"Could not load {fname}: {e}")

# Evaluate
results = []
plt.figure(figsize=(9,6))

for name, pipe in model_pipes.items():
    if pipe is None:
        print(f"Skipping {name} (no pipeline found).")
        continue

    # ensure pipeline is fitted
    try:
        _ = pipe.predict(X_test[:1])
    except Exception:
        print(f"Pipeline for {name} not fitted — fitting now on X_train/y_train.")
        pipe.fit(X_train, y_train)

    # predictions
    y_pred = pipe.predict(X_test)

    # score for ROC: prefer predict_proba, then decision_function, otherwise fallback to labels
    y_score = None
    if hasattr(pipe, "predict_proba"):
        try:
            y_score = pipe.predict_proba(X_test)[:, 1]
        except Exception:
            y_score = None
    if y_score is None and hasattr(pipe, "decision_function"):
        try:
            y_score = pipe.decision_function(X_test)
        except Exception:
            y_score = None
    if y_score is None:
        y_score = y_pred  # degenerate fallback

    # compute metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        roc = roc_auc_score(y_test, y_score)
    except Exception:
        roc = np.nan

    results.append({
        'model': name,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': roc
    })

    # plot ROC
    try:
        fpr, tpr, _ = roc_curve(y_test, y_score)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc:.3f})")
    except Exception:
        pass

# finalize ROC plot
plt.plot([0,1],[0,1], linestyle='--', color='k', linewidth=0.8)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves — Model Comparison')
plt.legend(loc='lower right', fontsize='small')
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()

# results DataFrame
df_res = pd.DataFrame(results)
# round for display
df_res_display = df_res.copy()
df_res_display[['accuracy','precision','recall','f1','roc_auc']] = df_res_display[['accuracy','precision','recall','f1','roc_auc']].round(4)
df_res_sorted = df_res_display.sort_values(by='roc_auc', ascending=False).reset_index(drop=True)

print("\nModel comparison sorted by ROC-AUC:")
display(df_res_sorted)

# Best by each metric
best_by_acc = df_res.loc[df_res['accuracy'].idxmax()]
best_by_prec = df_res.loc[df_res['precision'].idxmax()]
best_by_rec = df_res.loc[df_res['recall'].idxmax()]
best_by_f1 = df_res.loc[df_res['f1'].idxmax()]
best_by_auc = df_res.loc[df_res['roc_auc'].idxmax()]

print("\nBest models by metric:")
print(f" - Accuracy : {best_by_acc['model']} (accuracy={best_by_acc['accuracy']:.4f})")
print(f" - Precision: {best_by_prec['model']} (precision={best_by_prec['precision']:.4f})")
print(f" - Recall   : {best_by_rec['model']} (recall={best_by_rec['recall']:.4f})")
print(f" - F1-score : {best_by_f1['model']} (f1={best_by_f1['f1']:.4f})")
print(f" - ROC-AUC  : {best_by_auc['model']} (roc_auc={best_by_auc['roc_auc']:.4f})")



# In[37]:


#  code to print the best model by ROC-AUC ---
best_model_name = df_res_sorted.iloc[0]['model']
best_model_auc  = df_res_sorted.iloc[0]['roc_auc']

print(f"Best model by ROC-AUC: {best_model_name} (AUC={best_model_auc:.4f})")


# In[ ]:





# In[ ]:





# In[ ]:




