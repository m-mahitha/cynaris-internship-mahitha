import numpy as np
array_1d = np.array([10,20,30,40])
print(array_1d)
print(array_1d.shape)

array_2d = np.array([
    [10,20,30],
    [40,50,60]
])
print(array_2d)
print(array_2d.shape)

array_3d = np.around([
    [
        [1,2],
        [3,4]
    ],
    [
        [5,6],
        [7,8]
    ]

])
print(array_3d)
print(array_3d.shape)

#Broadcasting
numbers=np.array([10,20,30,40])
result = numbers + 5
print("Broadcasting:")
print(result)

#Vectorized operations
numbers = np.array([10,20,30,40])
double = numbers * 2
square = numbers ** 2
print("\nVectorized operations:")
print("Double:", double)
print("Square:", square)

#Matrix Multiplication
matrix_a=np.array([
    [1,2],
    [3,4]
])
matrix_b=np.array([
    [5,6],
    [7,8]
])
matrix_result = matrix_a @ matrix_b
print("\nMatrix Multiplication:")
print(matrix_result)

# Task 3: Statistics on Healthcare Dataset

import pandas as pd

# Load the healthcare dataset
df = pd.read_csv("aiml_healthcare_symptoms.csv")

# Convert Age and BMI columns to NumPy arrays
age = df["Age"].to_numpy()
bmi = df["BMI"].to_numpy()

# Calculate mean
age_mean = np.mean(age)
bmi_mean = np.mean(bmi)

# Calculate standard deviation
age_std = np.std(age)
bmi_std = np.std(bmi)

# Calculate correlation
correlation = np.corrcoef(age, bmi)[0, 1]

# Display results
print("\nHealthcare Dataset Statistics:")
print("Age Mean:", age_mean)
print("Age Standard Deviation:", age_std)
print("BMI Mean:", bmi_mean)
print("BMI Standard Deviation:", bmi_std)
print("Correlation between Age and BMI:", correlation)

# ==========================================
# DAY 2 - PANDAS DATA MANIPULATION
# ==========================================

import pandas as pd
import os

# Task 1: Load and inspect the dataset

df = pd.read_csv("Bangalore_AQI_Dataset.csv")

print("\n========== DAY 2: DATASET INSPECTION ==========")

print("\nDataset Shape:")
print(df.shape)

print("\nData Types:")
print(df.dtypes)

print("\nFirst 10 Rows:")
print(df.head(10))

# Task 2: Filter

# Select records where AQI is greater than 200
high_aqi = df[df["AQI"] > 200]

print("\n========== FILTER: AQI > 200 ==========")
print(high_aqi.head(10))

# Task 2: GroupBy

# Calculate average AQI for each city
grouped_data = df.groupby("City")["AQI"].mean()

print("\n========== GROUPBY: AVERAGE AQI BY CITY ==========")
print(grouped_data)

# Task 2: Merge

# Create a small City-State lookup DataFrame
city_state = pd.DataFrame({
    "City": df["City"].unique(),
    "State": "Karnataka"
})

# Merge the two DataFrames using the City column
merged_data = pd.merge(
    df,
    city_state,
    on="City",
    how="left"
)

print("\n========== MERGE: CITY + STATE ==========")
print(merged_data.head(10))

# Task 2: Pivot Table

# Calculate average AQI for each city using a pivot table
pivot_data = pd.pivot_table(
    df,
    values="AQI",
    index="City",
    aggfunc="mean"
)

print("\n========== PIVOT TABLE: AVERAGE AQI ==========")
print(pivot_data)

# Task 3: Export to CSV and Parquet

# Save the DataFrame as CSV
df.to_csv(
    "bangalore_aqi_cleaned.csv",
    index=False
)

# Save the DataFrame as Parquet
df.to_parquet(
    "bangalore_aqi_cleaned.parquet",
    index=False
)

# Compare file sizes

csv_size = os.path.getsize(
    "bangalore_aqi_cleaned.csv"
)

parquet_size = os.path.getsize(
    "bangalore_aqi_cleaned.parquet"
)

print("\n========== FILE SIZE COMPARISON ==========")
print("CSV size:", csv_size, "bytes")
print("Parquet size:", parquet_size, "bytes")

print("\nDay 2 tasks completed successfully!")

# ==========================================
# DAY 3 - DATA LOADING, CLEANING & INSPECTION
# ==========================================

import pandas as pd
import numpy as np

# Task 1: Load the healthcare dataset

df = pd.read_csv("aiml_healthcare_symptoms.csv")

print("\n========== DAY 3: DATA INSPECTION ==========")

print("\nOriginal Dataset Shape:")
print(df.shape)

print("\nDataset Columns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nFirst 5 Rows:")
print(df.head())

# Task 2: Check missing values

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# Task 3: Check duplicate rows

duplicate_count = df.duplicated().sum()

print("\n========== DUPLICATE ROWS ==========")
print("Number of duplicate rows:", duplicate_count)

# Task 4: Remove duplicate rows

df = df.drop_duplicates()

print("\nShape After Removing Duplicates:")
print(df.shape)

# Task 5: Identify numerical and categorical columns

numerical_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

categorical_columns = df.select_dtypes(
    exclude=np.number
).columns.tolist()

print("\n========== NUMERICAL COLUMNS ==========")
print(numerical_columns)

print("\n========== CATEGORICAL COLUMNS ==========")
print(categorical_columns)

# Task 6: Handle missing values

# Fill missing numerical values with the median
for column in numerical_columns:
    if df[column].isnull().any():
        df[column] = df[column].fillna(
            df[column].median()
        )

# Fill missing categorical values with the mode
for column in categorical_columns:
    if df[column].isnull().any():
        df[column] = df[column].fillna(
            df[column].mode()[0]
        )

# Task 7: Verify the cleaned dataset

print("\n========== MISSING VALUES AFTER CLEANING ==========")
print(df.isnull().sum())

print("\nFinal Dataset Shape:")
print(df.shape)

print("\nDuplicate Rows After Cleaning:")
print(df.duplicated().sum())

# Task 8: Basic validation tests

assert df.shape[0] > 0
assert df.isnull().sum().sum() == 0
assert df.duplicated().sum() == 0

print("\n========== VALIDATION ==========")
print("All data-cleaning tests passed!")

# Task 9: Save the cleaned dataset

df.to_csv(
    "aiml_healthcare_symptoms_cleaned.csv",
    index=False
)

print("\nCleaned dataset saved as:")
print("aiml_healthcare_symptoms_cleaned.csv")

print("\n========== DAY 3 COMPLETED ==========")

# ==========================================
# DAY 4 - EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the cleaned healthcare dataset

df = pd.read_csv("aiml_healthcare_symptoms_cleaned.csv")

print("\n==========================================")
print("DAY 4 - EXPLORATORY DATA ANALYSIS")
print("==========================================")

# Task 1: Dataset Information

print("\n========== DESCRIBE ==========")
print(df.describe())

print("\n========== DATASET INFO ==========")
df.info()

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# Identify numerical and categorical columns

numerical_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

categorical_columns = df.select_dtypes(
    exclude=np.number
).columns.tolist()

print("\n========== NUMERICAL COLUMNS ==========")
print(numerical_columns)

print("\n========== CATEGORICAL COLUMNS ==========")
print(categorical_columns)

# Task 2A: Distribution of each numerical column

print("\n========== NUMERICAL DISTRIBUTIONS ==========")

for column in numerical_columns:

    plt.figure(figsize=(7, 5))

    plt.hist(
        df[column].dropna(),
        bins=20
    )

    plt.title("Distribution of " + column)
    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.tight_layout()

    # Save the plot
    filename = "distribution_" + column + ".png"
    plt.savefig(filename)

    plt.show()

# Task 2B: Correlation Heatmap

print("\n========== CORRELATION HEATMAP ==========")

correlation_matrix = df[numerical_columns].corr()

plt.figure(figsize=(10, 8))

plt.imshow(
    correlation_matrix,
    cmap="coolwarm",
    interpolation="nearest"
)

plt.colorbar()

plt.xticks(
    range(len(numerical_columns)),
    numerical_columns,
    rotation=90
)

plt.yticks(
    range(len(numerical_columns)),
    numerical_columns
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("correlation_heatmap.png")

plt.show()

# Task 2C: Top-10 category counts

print("\n========== TOP-10 CATEGORY COUNTS ==========")

for column in categorical_columns:

    print("\nTop categories for:", column)

    top_categories = df[column].value_counts().head(10)

    print(top_categories)

    plt.figure(figsize=(8, 5))

    top_categories.plot(
        kind="bar"
    )

    plt.title("Top 10 Categories - " + column)
    plt.xlabel(column)
    plt.ylabel("Count")

    plt.xticks(rotation=45)

    plt.tight_layout()

    filename = "top10_" + column + ".png"
    plt.savefig(filename)

    plt.show()

# Task 3: Basic EDA observations

print("\n========== EDA OBSERVATIONS ==========")

print("1. The dataset contains", df.shape[0], "rows and", df.shape[1], "columns.")

print(
    "2. The dataset contains",
    len(numerical_columns),
    "numerical columns and",
    len(categorical_columns),
    "categorical columns."
)

print(
    "3. Total missing values:",
    df.isnull().sum().sum()
)

print(
    "4. Total duplicate rows:",
    df.duplicated().sum()
)

# Find the strongest correlation
correlation_matrix_no_diagonal = correlation_matrix.copy()

for column in correlation_matrix_no_diagonal.columns:
    correlation_matrix_no_diagonal.loc[column, column] = np.nan

strongest_correlation = (
    correlation_matrix_no_diagonal
    .abs()
    .stack()
    .idxmax()
)

strongest_value = (
    correlation_matrix_no_diagonal
    .loc[strongest_correlation]
)

print(
    "5. Strongest numerical correlation:",
    strongest_correlation[0],
    "and",
    strongest_correlation[1],
    "=",
    round(strongest_value, 3)
)

# Task 4: Save correlation matrix

correlation_matrix.to_csv(
    "correlation_matrix.csv"
)

print("\nCorrelation matrix saved as:")
print("correlation_matrix.csv")

# Day 4 Practical Work Completed

print("\n==========================================")
print("DAY 4 PRACTICAL TASKS COMPLETED")
print("==========================================")

# Day 4: EDA Observations and Narrative

# Five observations:
# 1. The cleaned dataset contains 800 rows and 17 columns.
# 2. There are no missing values in the cleaned dataset.
# 3. There are no duplicate rows after cleaning.
# 4. Numerical columns include age, BMI, blood pressure,
#    SpO2, glucose, and symptom-related fields.
# 5. Categorical fields include Patient_ID, Gender, and Diagnosis.
#    Numerical distributions and correlations should be checked
#    to identify unusual values and relationships.

# EDA Narrative:
# The healthcare dataset contains 800 records and 17 columns.
# Exploratory Data Analysis was performed to understand its
# structure, data types, distributions, and numerical relationships.
# The dataset information and descriptive statistics help identify
# the range, average, and variation of numerical measurements.
# The missing-value check returned zero, and no duplicate rows
# remained after cleaning. Numerical fields include age, BMI,
# blood pressure measurements, oxygen saturation, glucose, and
# symptom-related variables. Patient ID, gender, and diagnosis
# are categorical fields in the current analysis.
#
# Distribution plots were generated to inspect the spread of
# numerical variables, while a correlation heatmap was created
# to examine relationships between numerical features. These
# visualizations can help identify unusual values and possible
# associations that may need further investigation. However,
# correlation does not prove causation. Patient IDs should not
# be treated as meaningful medical predictors simply because
# they are stored as numbers. The diagnosis field should also
# be examined separately because it represents the outcome
# category. Before using this dataset for machine learning,
# the values should be checked against their definitions and
# plausible medical ranges. Any suspicious values should be
# investigated rather than removed automatically. Overall,
# the dataset appears complete after cleaning, but its medical
# validity and suitability for modelling require further checks.

# Week 1: NumPy, Pandas, data cleaning, and exploratory data analysis completed.

# Week 1 Final Checklist
# - NumPy and Pandas practice
# - Data cleaning and preprocessing
# - Exploratory Data Analysis (EDA)
# - Data visualization
# - Results and observations documented