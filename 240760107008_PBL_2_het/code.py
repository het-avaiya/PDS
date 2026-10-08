import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# Configure visualization styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150
[ ]:
print("====================================================================")
print("--- Unit 1 & 2: File Handling with Exception Handling ---")
print("====================================================================")
dataset_filename = 'college_placement_dataset_500.csv'
try:
if not os.path.exists(dataset_filename):
raise FileNotFoundError(f"Target file '{dataset_filename}' was not found in directory.")
df = pd.read_csv(dataset_filename)
print(f"[SUCCESS] File '{dataset_filename}' successfully loaded.")
print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
except FileNotFoundError as fnf_error:
print(f"[ERROR] File Not Found: {fnf_error}")
raise
except pd.errors.EmptyDataError:
print("[ERROR] The specified CSV file is empty.")
raise
except Exception as general_error:
print(f"[ERROR] Unexpected error occurred: {general_error}")
raise
finally:
print("[STATUS] File ingestion pipeline execution finished.\n")
print("--- First 5 Records ---")
print(df.head())
print("\n--- Dataset Schema & Data Types ---")
df.info()
[ ]:
print("\n====================================================================")
print("--- Unit 6: Data Cleaning, Outliers & Min-Max Normalization ---")
print("====================================================================")
# 1. Missing Values and Duplicates Check
print("Missing values per column:")
print(df.isnull().sum())
duplicate_count = df.duplicated().sum()
print(f"\nDuplicate records detected: {duplicate_count}")
if duplicate_count > 0:
df.drop_duplicates(inplace=True)
print("Duplicates removed.")
# 2. Outlier Detection using Interquartile Range (IQR) Method
print("\n--- Outlier Detection via IQR ---")
q1 = df['CGPA'].quantile(0.25)
q3 = df['CGPA'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = df[(df['CGPA'] < lower_bound) | (df['CGPA'] > upper_bound)]
print(f"CGPA Q1: {q1:.2f}, Q3: {q3:.2f}, IQR: {iqr:.2f}")
print(f"Valid Non-Outlier Range: [{lower_bound:.2f}, {upper_bound:.2f}]")
print(f"Statistical Outliers in CGPA: {len(outliers)}")
# 3. Data Normalization (Min-Max Scaling to [0, 1])
print("\n--- Data Normalization (Min-Max Scaling) ---")
scaler = MinMaxScaler()
scaled_cols = ['CGPA', 'Aptitude_Score', 'Technical_Skills', 'Communication_Score']
normalized_matrix = scaler.fit_transform(df[scaled_cols])
df_normalized = pd.DataFrame(normalized_matrix, columns=[f"{c}_norm" for c in scaled_cols])
print("Sample Normalized Attributes (First 5 Rows):")
print(df_normalized.head())
print("\n====================================================================")
print("--- Unit 3: Feature Engineering & Placement Target Formulation ---")
print("====================================================================")
# Calculate composite placement score
composite_score = (
(df['CGPA'] / 10.0) * 35 +
(df['Aptitude_Score'] / 100.0) * 25 +
(df['Technical_Skills'] / 10.0) * 20 +
(df['Communication_Score'] / 100.0) * 15 +
(df['Internship'].apply(lambda x: 1 if x == 'Yes' else 0)) * 5
)
df['Composite_Score'] = composite_score.round(2)
# Eligibility: CGPA >= 6.0 and Backlogs <= 1
is_eligible = (df['CGPA'] >= 6.0) & (df['Backlogs'] <= 1)
df['Placement_Status'] = np.where(
is_eligible & (df['Composite_Score'] >= 60.0),
'Placed',
'Not Placed'
)
# Estimated package in LPA for placed candidates
df['Salary_LPA'] = np.where(
df['Placement_Status'] == 'Placed',
np.round(
3.5 +
(df['CGPA'] - 6.0) * 1.2 +
(df['Technical_Skills'] / 10.0) * 2.5 +
(df['Projects'] * 0.4),
2
),
0.0
)
print("Placement Outcome Distribution:")
print(df['Placement_Status'].value_counts())
print("\nPlacement Outcome Percentage (%):")
print((df['Placement_Status'].value_counts(normalize=True) * 100).round(2))
[ ]:
print("\n====================================================================")
print("--- Unit 4 & 5: SciPy Statistics & Hypothesis Testing ---")
print("====================================================================")
# 1. SciPy Measures of Shape & Symmetry
cgpa_skewness = stats.skew(df['CGPA'])
cgpa_kurtosis = stats.kurtosis(df['CGPA'])
print(f"CGPA Skewness: {cgpa_skewness:.4f} (Near 0 -> symmetric distribution)")
print(f"CGPA Kurtosis: {cgpa_kurtosis:.4f}")
hypothesized_mean = 7.50
sample_mean = df['CGPA'].mean()
sample_std = df['CGPA'].std()
t_statistic, p_value = stats.ttest_1samp(df['CGPA'], popmean=hypothesized_mean)
print(f"\nOne-Sample t-Test against Hypothesized Mean ({hypothesized_mean}):")
print(f"Sample Mean: {sample_mean:.4f} | Sample Std: {sample_std:.4f}")
print(f"Calculated t-statistic: {t_statistic:.4f}")
print(f"p-value: {p_value:.4e}")
alpha = 0.05
if p_value < alpha:
print(f"Decision: Reject H0 at alpha={alpha}. Mean CGPA is significantly greater than 7.50.")
else:
print(f"Decision: Fail to reject H0 at alpha={alpha}.")
# 3. Department-wise Placement Analysis
print("\nDepartment-wise Placement Rates (%):")
dept_analysis = (
df.groupby('Department')['Placement_Status']
.value_counts(normalize=True)
.unstack()
.fillna(0) * 100
)
print(dept_analysis.round(2))
[ ]:
print("\n====================================================================")
print("--- Unit 7: Generating 6 Statistical Visualizations ---")
print("====================================================================")
# Figure 1: Distribution of Academic CGPA
plt.figure(figsize=(8, 5))
sns.histplot(df['CGPA'], kde=True, color='#2b5c8f', bins=20)
plt.title('Figure 1: Distribution of Student CGPA', fontsize=14, weight='bold')
plt.xlabel('Cumulative Grade Point Average (CGPA)', fontsize=11)
plt.ylabel('Frequency', fontsize=11)
plt.tight_layout()
plt.savefig('fig1_cgpa_distribution.png')
plt.close()
# Figure 2: CGPA vs Placement Status Boxplot
plt.figure(figsize=(7, 5))
sns.boxplot(x='Placement_Status', y='CGPA', data=df, palette='Set2')
plt.title('Figure 2: CGPA Spread by Placement Status', fontsize=14, weight='bold')
plt.xlabel('Placement Outcome', fontsize=11)
plt.ylabel('CGPA', fontsize=11)
plt.tight_layout()
plt.savefig('fig2_cgpa_placement_box.png')
plt.close()
# Figure 3: Department-wise Placement Comparison
plt.figure(figsize=(10, 5))
sns.countplot(x='Department', hue='Placement_Status', data=df, palette='viridis')
plt.title('Figure 3: Placement Outcomes Across Engineering Departments', fontsize=14, weight='bold')
plt.xlabel('Engineering Branch', fontsize=11)
plt.ylabel('Student Count', fontsize=11)
plt.xticks(rotation=25)
plt.legend(title='Status')
plt.tight_layout()
plt.savefig('fig3_dept_placement_status.png')
plt.close()
# Figure 4: Correlation Heatmap
plt.figure(figsize=(10, 8))
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Figure 4: Pearson Correlation Heatmap of Student Attributes', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig('fig4_correlation_heatmap.png')
plt.close()
# Figure 5: Aptitude vs Communication by Placement Status
plt.figure(figsize=(8, 6))
sns.scatterplot(
x='Aptitude_Score',
y='Communication_Score',
hue='Placement_Status',
style='Placement_Status',
data=df,
palette='coolwarm',
s=70,
alpha=0.85
)
plt.title('Figure 5: Aptitude vs. Communication Score by Placement Status', fontsize=14, weight='bold')
plt.xlabel('Aptitude Test Score (out of 100)', fontsize=11)
plt.ylabel('Communication Score (out of 100)', fontsize=11)
plt.tight_layout()
plt.savefig('fig5_aptitude_vs_communication.png')
plt.close()
# Figure 6: Impact of Internships on Placement
plt.figure(figsize=(6, 5))
sns.countplot(x='Internship', hue='Placement_Status', data=df, palette='pastel')
plt.title('Figure 6: Impact of Prior Internship on Placement Success', fontsize=14, weight='bold')
plt.xlabel('Completed Internship Experience', fontsize=11)
plt.ylabel('Number of Students', fontsize=11)
plt.tight_layout()
plt.savefig('fig6_internship_impact.png')
plt.close()
print("Saved all 6 figures to current directory:")
print(" - fig1_cgpa_distribution.png")
print(" - fig2_cgpa_placement_box.png")
print(" - fig3_dept_placement_status.png")
print(" - fig4_correlation_heatmap.png")
print(" - fig5_aptitude_vs_communication.png")
print(" - fig6_internship_impact.png")
print("\n====================================================================")
print("--- Supervised ML: Random Forest Classifier ---")
print("====================================================================")
features = [
'CGPA',
'Backlogs',
'Projects',
'Technical_Skills',
'Aptitude_Score',
'Communication_Score',
'Certifications'
]
X = df[features]
y = df['Placement_Status'].apply(lambda x: 1 if x == 'Placed' else 0)
# 80/20 Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Training instances: {len(X_train)} | Testing instances: {len(X_test)}")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
rf_model.fit(X_train, y_train)
# Predictions & evaluation
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Test Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Placed', 'Placed']))
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("\nFeature Importance Breakdown:")
importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
print(importances.round(4))
print("\n====================================================================")
print("Execution Finished Successfully!")
print("====================================================================")
