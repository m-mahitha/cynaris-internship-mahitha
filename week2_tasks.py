# WEEK 2 - DAY 1: FEATURE ENGINEERING

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import (
    LabelEncoder, OneHotEncoder, OrdinalEncoder,
    StandardScaler, MinMaxScaler, RobustScaler
)
from sklearn.feature_selection import SelectKBest, f_classif

os.makedirs("week2_day1_outputs", exist_ok=True)
out = "week2_day1_outputs/"
df = pd.read_csv("aiml_healthcare_symptoms_cleaned.csv")

cat = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
num = df.select_dtypes(include=np.number).columns.tolist()
scale_cols = [c for c in num if "id" not in c.lower()]

print("Shape:", df.shape)
print("Categorical:", cat)
print("Numerical:", num)

# 1. Encoding methods
if cat:
    col = cat[0]
    original = df[col].astype(str)

    le = LabelEncoder()
    pd.DataFrame({
        "Original": original,
        "Encoded": le.fit_transform(original)
    }).drop_duplicates().sort_values("Encoded").to_csv(
        out + "label_encoding.csv", index=False
    )

    oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    onehot = pd.DataFrame(
        oh.fit_transform(df[[col]].astype(str)),
        columns=oh.get_feature_names_out([col])
    )
    onehot.to_csv(out + "onehot_encoding.csv", index=False)

    oe = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1
    )
    pd.DataFrame({
        "Original": original,
        "Encoded": oe.fit_transform(df[[col]].astype(str)).ravel().astype(int)
    }).drop_duplicates().sort_values("Encoded").to_csv(
        out + "ordinal_encoding.csv", index=False
    )

print("Encoding files saved.")

# 2. Scaling methods
if scale_cols:
    X = df[scale_cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    scalers = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler()
    }
    results = {name: pd.DataFrame(
        scaler.fit_transform(X), columns=scale_cols
    ) for name, scaler in scalers.items()}

    for name, data in results.items():
        data.to_csv(out + name + "_scaled.csv", index=False)

    # Plot original and scaled distributions for up to 3 columns
    for col in scale_cols[:3]:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for ax, data, title in zip(
            axes, [X] + list(results.values()),
            ["Original", "StandardScaler", "MinMaxScaler", "RobustScaler"]
        ):
            ax.hist(data[col], bins=20)
            ax.set_title(title)
            ax.set_xlabel(col)
        plt.tight_layout()
        safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in col)
        plt.savefig(out + f"scaling_{safe}.png", dpi=150)
        plt.show()
        plt.close()

# 3. Select top 5 numerical features for Diagnosis
target = next((c for c in df.columns if c.lower() == "diagnosis"), None)

if target:
    features = [c for c in scale_cols if c != target]
    if features:
        X = df[features].replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        y = df[target].astype(str)

        selector = SelectKBest(f_classif, k=min(5, len(features)))
        selector.fit(X, y)

        scores = pd.DataFrame({
            "Feature": features,
            "F_Score": selector.scores_,
            "P_Value": selector.pvalues_,
            "Selected": selector.get_support()
        }).sort_values("F_Score", ascending=False)

        scores.to_csv(out + "feature_selection_scores.csv", index=False)
        print("\nTop features:\n", scores[scores["Selected"]].to_string(index=False))
else:
    print("Diagnosis target column not found.")

# 4. Summary
summary = f"""WEEK 2 - DAY 1: FEATURE ENGINEERING
Dataset shape: {df.shape}
Categorical columns: {cat}
Numerical columns: {num}

Encoding:
- LabelEncoder: integer codes; may imply a false order.
- OneHotEncoder: binary columns; can increase dimensionality.
- OrdinalEncoder: use when a meaningful category order is defined.

Scaling:
- StandardScaler: mean 0 and standard deviation 1.
- MinMaxScaler: scales values to 0-1.
- RobustScaler: uses median and interquartile range.

Feature selection:
SelectKBest ranks numerical predictors by their ANOVA F-score.
Association does not prove causation.
"""
with open(out + "day1_summary.txt", "w", encoding="utf-8") as f:
    f.write(summary)

print("\nDay 1 completed. Results saved in", out)


# WEEK 2 - DAY 2 - TASK 1: FEATURE SCALING

from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, Normalizer
)

# Load cleaned dataset
df2 = pd.read_csv("aiml_healthcare_symptoms_cleaned.csv")

# Select numerical columns, excluding ID-like columns
num_cols = [
    c for c in df2.select_dtypes(include=np.number).columns
    if "id" not in c.lower()
]

X = df2[num_cols].replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

# Apply four scaling methods
scalers = {
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler(),
    "RobustScaler": RobustScaler(),
    "Normalizer": Normalizer()
}

scaled_data = {
    name: pd.DataFrame(
        scaler.fit_transform(X), columns=num_cols
    )
    for name, scaler in scalers.items()
}

# Save scaled datasets
os.makedirs("week2_day2_outputs", exist_ok=True)

for name, data in scaled_data.items():
    data.to_csv(
        f"week2_day2_outputs/{name.lower()}_data.csv",
        index=False
    )

# Plot original and scaled distributions for up to 3 columns
for col in num_cols[:3]:
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))

    datasets = [("Original", X)] + list(scaled_data.items())

    for ax, (name, data) in zip(axes, datasets):
        ax.hist(data[col], bins=20)
        ax.set_title(name)
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(
        f"week2_day2_outputs/scaling_{col}.png",
        dpi=150,
        bbox_inches="tight"
    )
    plt.show()
    plt.close()

print("Task 1 completed!")
print("Scaling results and plots saved in week2_day2_outputs.")


# WEEK 2 - DAY 2 - TASK 2: SELECTKBEST

from sklearn.feature_selection import SelectKBest, f_classif, chi2

target = next((c for c in df2.columns if c.lower() == "diagnosis"), None)
if target is None:
    raise ValueError("Diagnosis target column not found.")

features = [c for c in num_cols if c != target]
X = df2[features].replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())
y = df2[target].astype(str)

f_test = SelectKBest(f_classif, k="all").fit(X, y)
X_chi = X - X.min()
chi_test = SelectKBest(chi2, k="all").fit(X_chi, y)

rankings = pd.DataFrame({
    "Feature": features,
    "F_Score": f_test.scores_,
    "F_Rank": pd.Series(f_test.scores_).rank(ascending=False).values,
    "Chi2_Score": chi_test.scores_,
    "Chi2_Rank": pd.Series(chi_test.scores_).rank(ascending=False).values
})
rankings.to_csv("week2_day2_outputs/selectkbest_rankings.csv", index=False)
print(rankings.sort_values("F_Rank").to_string(index=False))

print("""
f_classif compares feature means across target classes.
chi2 measures association between non-negative feature values and classes.
Their rankings differ because they use different statistical tests.
""")


# WEEK 2 - DAY 2 - TASK 3: RFE FEATURE SELECTION

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rfe = RFE(
    LogisticRegression(max_iter=2000),
    n_features_to_select=min(5, len(features))
)
rfe.fit(X_train, y_train)

rfe_results = pd.DataFrame({
    "Feature": features,
    "Selected": rfe.support_,
    "RFE_Rank": rfe.ranking_
}).sort_values("RFE_Rank")

rfe_results.to_csv("week2_day2_outputs/rfe_feature_selection.csv", index=False)
print("RFE results:")
print(rfe_results.to_string(index=False))
print("Eliminated features:",
      rfe_results.loc[~rfe_results["Selected"], "Feature"].tolist())

model = LogisticRegression(max_iter=2000)
model.fit(rfe.transform(X_train), y_train)
pred = model.predict(rfe.transform(X_test))
print("Test accuracy:", round(accuracy_score(y_test, pred), 4))


# WEEK 2 - DAY 2 - TASK 4: SUMMARY

summary = """
Feature Scaling:
StandardScaler standardizes features.
MinMaxScaler maps values to a fixed range.
RobustScaler uses median and interquartile range.
Normalizer scales each row.

Feature Selection:
SelectKBest compares ANOVA F-scores and chi-square scores.
RFE selects features recursively using Logistic Regression.
Rankings are model- and dataset-dependent.
"""

with open("week2_day2_outputs/day2_summary.txt", "w", encoding="utf-8") as f:
    f.write(summary)

print("Day 2 tasks completed. Check week2_day2_outputs.")


# ==========================================
# WEEK 2 - DAY 3: HANDLING IMBALANCED DATA
# ==========================================

def run_day3():

    import os
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from imblearn.over_sampling import SMOTE

    os.makedirs("week2_day3_outputs", exist_ok=True)

    # TASK 1: Load data and analyze class imbalance
    df = pd.read_csv("aiml_healthcare_symptoms_cleaned.csv")

    target = "Diagnosis"

    if target not in df.columns:
        raise ValueError(
            f"Target '{target}' not found. Check your dataset columns."
        )

    X = df.drop(columns=[target]).copy()
    y = df[target].astype(str)

    # Convert categorical input features into numeric columns
    X = pd.get_dummies(X, drop_first=False)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([float("inf"), float("-inf")], float("nan"))
    X = X.dropna(axis=1, how="all")
    X = X.fillna(X.median())

    # Encode target labels
    encoder = LabelEncoder()
    y = pd.Series(encoder.fit_transform(y), name=target)

    before_counts = y.value_counts().sort_index()

    print("\nTASK 1: Class Distribution Before SMOTE")
    print(before_counts)

    before_counts.to_csv(
        "week2_day3_outputs/class_counts_before.csv",
        header=["Count"]
    )

    # Split before applying SMOTE to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # TASK 2: Apply SMOTE to training data only
    # SMOTE requires enough minority-class samples for its neighbours.
    min_class_count = y_train.value_counts().min()

    if min_class_count < 2:
        raise ValueError(
            "SMOTE needs at least 2 training samples in every class."
        )

    smote = SMOTE(
        random_state=42,
        k_neighbors=min(5, min_class_count - 1)
    )

    X_train_smote, y_train_smote = smote.fit_resample(
        X_train, y_train
    )

    print("\nTASK 2: Class Distribution After SMOTE")
    print(y_train_smote.value_counts().sort_index())

    # TASK 3: Compare, validate, and save outputs
    after_counts = y_train_smote.value_counts().sort_index()

    comparison = pd.DataFrame({
        "Before_SMOTE_Training": y_train.value_counts(),
        "After_SMOTE_Training": after_counts
    }).fillna(0).astype(int)

    comparison.to_csv(
        "week2_day3_outputs/smote_comparison.csv"
    )

    # Confirm training classes are balanced
    assert after_counts.nunique() == 1, (
        "SMOTE training classes are not balanced."
    )

    # Save balanced training data and the original test data
    X_train_smote.to_csv(
        "week2_day3_outputs/X_train_smote.csv",
        index=False
    )
    y_train_smote.to_csv(
        "week2_day3_outputs/y_train_smote.csv",
        index=False
    )
    X_test.to_csv(
        "week2_day3_outputs/X_test.csv",
        index=False
    )
    y_test.to_csv(
        "week2_day3_outputs/y_test.csv",
        index=False
    )

    print("\nTASK 3: Validation Successful")
    print("SMOTE training classes are balanced.")
    print("Test data remains unchanged.")
    print("All output files saved in week2_day3_outputs.")
    print("\nWeek 2 Day 3 completed!")


# ==========================================
# DAY SELECTOR
# ==========================================
# Keep your existing Day 1 and Day 2 functions.
# Set day = 3 to run only Day 3.

day = 3

if day == 1:
    run_day1()
elif day == 2:
    run_day2()
elif day == 3:
    run_day3()
elif day == 4:
    run_day4()


# ==========================================
# WEEK 2 - DAY 4: TRAIN/TEST SPLIT & CROSS-VALIDATION
# ==========================================

def run_day4():

    import os
    import numpy as np
    import pandas as pd

    from sklearn.model_selection import (
        train_test_split, StratifiedKFold, cross_val_score
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report

    os.makedirs("week2_day4_outputs", exist_ok=True)

    # TASK 1: Load data and create train/test split
    df = pd.read_csv("aiml_healthcare_symptoms_cleaned.csv")
    target = "Diagnosis"

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")

    X = df.drop(columns=[target]).copy()
    y = df[target].astype(str)

    # Encode categorical features and remove invalid values
    X = pd.get_dummies(X, drop_first=False)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")

    # Split before fitting any preprocessing steps
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTASK 1: Train/Test Split")
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("\nTraining class distribution:")
    print(y_train.value_counts(normalize=True).round(3))
    print("\nTesting class distribution:")
    print(y_test.value_counts(normalize=True).round(3))

    # TASK 2: Cross-validation on training data only
    # Pipeline prevents preprocessing leakage across folds.
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000))
    ])

    smallest_class = y_train.value_counts().min()
    n_splits = min(5, int(smallest_class))

    if n_splits < 2:
        raise ValueError(
            "Not enough samples in each class for stratified cross-validation."
        )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=cv,
        scoring="accuracy"
    )

    print("\nTASK 2: Cross-Validation")
    print("Fold accuracy scores:", np.round(cv_scores, 4))
    print("Mean CV accuracy:", round(cv_scores.mean(), 4))
    print("CV standard deviation:", round(cv_scores.std(), 4))

    # TASK 3: Fit on training data and evaluate once on test data
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    test_accuracy = accuracy_score(y_test, predictions)

    print("\nTASK 3: Test Set Evaluation")
    print("Test accuracy:", round(test_accuracy, 4))
    print("\nClassification report:")
    report = classification_report(
        y_test, predictions, zero_division=0
    )
    print(report)

    # Save results
    pd.DataFrame({
        "Fold": range(1, len(cv_scores) + 1),
        "Accuracy": cv_scores
    }).to_csv(
        "week2_day4_outputs/cross_validation_scores.csv",
        index=False
    )

    pd.DataFrame({
        "Metric": ["Mean CV Accuracy", "CV Std Dev", "Test Accuracy"],
        "Score": [cv_scores.mean(), cv_scores.std(), test_accuracy]
    }).to_csv(
        "week2_day4_outputs/model_summary.csv",
        index=False
    )

    with open(
        "week2_day4_outputs/classification_report.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write(report)

    # Basic validation checks
    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)

    print("\nAll Day 4 tasks completed successfully.")
    print("Outputs saved in week2_day4_outputs.")

# ==========================================
# SELECT WHICH DAY TO RUN
# ==========================================

day = 4

if day == 1:
    run_day1()
elif day == 2:
    run_day2()
elif day == 3:
    run_day3()
elif day == 4:
    run_day4()
elif day == 5:
    run_day5()


# ============================================================
# WEEK 2 - DAY 5
# End-to-End Preprocessing Pipeline using Titanic Dataset
# ============================================================

def run_day5():

    import os
    import pandas as pd
    import numpy as np

    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    print("\nWEEK 2 - DAY 5: END-TO-END PREPROCESSING PIPELINE")
    print("=" * 55)

    # --------------------------------------------------------
    # TASK 1: LOAD DATASET AND PERFORM BASIC EDA
    # --------------------------------------------------------

    print("\nTASK 1: LOAD DATASET AND PERFORM BASIC EDA")

    # Read the Titanic dataset from the Downloads folder
    dataset_path = r"C:\Users\DELL\Downloads\train.csv"

    if not os.path.exists(dataset_path):
        print("\nERROR: Dataset not found!")
        print("Please check that train.csv is in your Downloads folder.")
        print("Expected path:", dataset_path)
        return

    df = pd.read_csv(dataset_path)

    print("\nFirst five rows:")
    print(df.head())

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nDataset information:")
    df.info()

    print("\nMissing values in each column:")
    print(df.isnull().sum())

    print("\nStatistical summary:")
    print(df.describe())

    # --------------------------------------------------------
    # TASK 2: DATA CLEANING AND PREPROCESSING
    # --------------------------------------------------------

    print("\nTASK 2: DATA CLEANING AND PREPROCESSING")

    # Separate target and input features
    target = "Survived"

    if target not in df.columns:
        print("ERROR: The target column 'Survived' was not found.")
        return

    # Remove rows where the target value is missing, if any
    df = df.dropna(subset=[target]).copy()

    # Remove columns that will not be used for this pipeline
    columns_to_drop = [
        "PassengerId",
        "Name",
        "Ticket",
        "Cabin"
    ]

    X = df.drop(
        columns=[target] + [
            col for col in columns_to_drop if col in df.columns
        ]
    )

    y = df[target].astype(int)

    # Identify numerical and categorical columns
    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    print("\nNumerical features:")
    print(numeric_features)

    print("\nCategorical features:")
    print(categorical_features)

    # Numerical preprocessing:
    # Fill missing values with median and scale the features
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing:
    # Fill missing values with the most frequent value
    # and convert categories into numeric columns
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine both preprocessing pipelines
    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    # Split the data before fitting preprocessing.
    # This helps prevent data leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Learn preprocessing details only from training data
    X_train_processed = preprocessor.fit_transform(X_train)

    # Apply the same transformations to test data
    X_test_processed = preprocessor.transform(X_test)

    # Convert transformed arrays to DataFrames
    feature_names = preprocessor.get_feature_names_out()

    X_train_processed = pd.DataFrame(
        X_train_processed,
        columns=feature_names,
        index=X_train.index
    )

    X_test_processed = pd.DataFrame(
        X_test_processed,
        columns=feature_names,
        index=X_test.index
    )

    print("\nTraining features shape:")
    print(X_train_processed.shape)

    print("\nTesting features shape:")
    print(X_test_processed.shape)

    # --------------------------------------------------------
    # TASK 3: VALIDATE AND EXPORT PROCESSED DATA
    # --------------------------------------------------------

    print("\nTASK 3: VALIDATE AND EXPORT PROCESSED DATA")

    output_folder = "week2_day5_outputs"
    os.makedirs(output_folder, exist_ok=True)

    # Check for missing values after preprocessing
    print("\nMissing values in training features:")
    print(X_train_processed.isnull().sum().sum())

    print("\nMissing values in testing features:")
    print(X_test_processed.isnull().sum().sum())

    # Confirm that train and test features have the same columns
    print("\nDo training and testing columns match?")
    print(X_train_processed.columns.equals(
        X_test_processed.columns
    ))

    # Export processed features and target labels
    X_train_processed.to_csv(
        os.path.join(output_folder, "X_train_processed.csv"),
        index=False
    )

    X_test_processed.to_csv(
        os.path.join(output_folder, "X_test_processed.csv"),
        index=False
    )

    y_train.to_frame(name=target).to_csv(
        os.path.join(output_folder, "y_train.csv"),
        index=False
    )

    y_test.to_frame(name=target).to_csv(
        os.path.join(output_folder, "y_test.csv"),
        index=False
    )

    # Save a brief preprocessing summary
    summary = {
        "Original dataset rows": len(df),
        "Original dataset columns": len(df.columns),
        "Training rows": len(X_train_processed),
        "Testing rows": len(X_test_processed),
        "Processed feature count": len(feature_names),
        "Missing values in processed training data":
            int(X_train_processed.isnull().sum().sum()),
        "Missing values in processed testing data":
            int(X_test_processed.isnull().sum().sum())
    }

    pd.Series(summary, name="Value").to_csv(
        os.path.join(output_folder, "preprocessing_summary.csv"),
        header=True
    )

    print("\nPreprocessing summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nFiles saved in:", output_folder)
    print("1. X_train_processed.csv")
    print("2. X_test_processed.csv")
    print("3. y_train.csv")
    print("4. y_test.csv")
    print("5. preprocessing_summary.csv")

    print("\nDAY 5 COMPLETED SUCCESSFULLY!")


# ============================================================
# DAY SELECTOR
# Change the number to run a particular day.
# ============================================================

day = 5

if day == 1:
    run_day1()
elif day == 2:
    run_day2()
elif day == 3:
    run_day3()
elif day == 4:
    run_day4()
elif day == 5:
    run_day5()



   
   

    

   

   

    

    

   
       

    
   
