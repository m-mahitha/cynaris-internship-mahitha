
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def run_day1():
    print("\nWEEK 3 - DAY 1: LINEAR REGRESSION")

    # Create an output folder
    output_dir = "week3_day1_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Load a real housing dataset
    housing = fetch_california_housing(as_frame=True)
    X = housing.data
    y = housing.target

    print("\nDataset shape:", X.shape)
    print("Features:", list(X.columns))
    print("Target: Median house value (in units of $100,000)")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Linear Regression, Ridge and Lasso
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01, max_iter=10000)
    }

    results = []
    predictions = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions[name] = y_pred

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results.append({
            "Model": name,
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        })

        print(f"\n{name}")
        print("MSE:", round(mse, 4))
        print("RMSE:", round(rmse, 4))
        print("MAE:", round(mae, 4))
        print("R²:", round(r2, 4))

        if name == "Linear Regression":
            coefficients = pd.DataFrame({
                "Feature": X.columns,
                "Coefficient": model.coef_
            })
            print("\nLinear Regression coefficients:")
            print(coefficients.to_string(index=False))
            coefficients.to_csv(
                os.path.join(output_dir, "linear_regression_coefficients.csv"),
                index=False
            )
            intercept = model.intercept_
            print("Intercept:", round(intercept, 4))

    # Save model comparison table
    results_df = pd.DataFrame(results)
    print("\nModel Comparison:")
    print(results_df.round(4).to_string(index=False))
    results_df.to_csv(
        os.path.join(output_dir, "model_comparison.csv"),
        index=False
    )

    # Plot 1: Predicted vs Actual
    y_pred_linear = predictions["Linear Regression"]

    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, y_pred_linear, alpha=0.3)
    lower = min(y_test.min(), y_pred_linear.min())
    upper = max(y_test.max(), y_pred_linear.max())
    plt.plot([lower, upper], [lower, upper], "r--")
    plt.xlabel("Actual House Value")
    plt.ylabel("Predicted House Value")
    plt.title("Linear Regression: Predicted vs Actual")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "predicted_vs_actual.png"),
        dpi=150
    )
    plt.close()

    # Plot 2: Residuals
    residuals = y_test - y_pred_linear

    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred_linear, residuals, alpha=0.3)
    plt.axhline(y=0, color="red", linestyle="--")
    plt.xlabel("Predicted House Value")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Linear Regression: Residual Plot")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "residuals.png"),
        dpi=150
    )
    plt.close()

    print(f"\nPlots and CSV files saved in: {output_dir}")


# Day selector: change this number when working on another day.
day = 1

if day == 1:
    run_day1()


def run_day2():
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report,
        ConfusionMatrixDisplay,
        roc_curve,
        roc_auc_score
    )

    print("\nWEEK 3 - DAY 2: LOGISTIC REGRESSION & CLASSIFICATION")

    output_dir = "week3_day2_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Load a real classification dataset
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target

    print("\nDataset shape:", X.shape)
    print("Target classes:", dict(enumerate(dataset.target_names)))
    print("Class counts:")
    print(y.value_counts().rename(index={
        0: dataset.target_names[0],
        1: dataset.target_names[1]
    }))

    # Split data while preserving class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scaling is fitted only on training data through the pipeline
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, random_state=42)
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Evaluation metrics
    results = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    }

    print("\nEvaluation Metrics:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred, target_names=dataset.target_names
    ))

    # Save evaluation results
    pd.DataFrame([results]).to_csv(
        os.path.join(output_dir, "classification_metrics.csv"),
        index=False
    )

    # Save coefficients
    fitted_model = model.named_steps["logisticregression"]
    coefficients = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": fitted_model.coef_[0]
    }).sort_values(
        by="Coefficient", key=abs, ascending=False
    )

    print("\nTop 10 coefficients by absolute value:")
    print(coefficients.head(10).to_string(index=False))

    coefficients.to_csv(
        os.path.join(output_dir, "logistic_regression_coefficients.csv"),
        index=False
    )

    # Plot 1: Confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=dataset.target_names,
        cmap="Blues"
    )
    plt.title("Logistic Regression - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "confusion_matrix.png"),
        dpi=150
    )
    plt.close()

    # Plot 2: ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {results['ROC-AUC']:.3f}")
    plt.plot([0, 1], [0, 1], "r--", label="Random classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Logistic Regression - ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "roc_curve.png"),
        dpi=150
    )
    plt.close()

    print(f"\nOutputs saved in: {output_dir}")


# Change the selector to run only the required day.
day = 2

if day == 1:
    run_day1()
elif day == 2:
    run_day2()


def run_day3():
    """Train, visualise, and compare Decision Tree and Random Forest models."""

    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        ConfusionMatrixDisplay,
    )

    # Create a folder to save Day 3 results
    output_dir = "week3_day3_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Load the dataset
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target

    print("\nWEEK 3 - DAY 3: DECISION TREES & RANDOM FORESTS")
    print("-" * 55)
    print("Dataset shape:", X.shape)
    print("Target classes:", dataset.target_names)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Train a default Decision Tree
    tree_default = DecisionTreeClassifier(random_state=42)
    tree_default.fit(X_train, y_train)

    # Tune the tree to reduce overfitting
    tree_tuned = DecisionTreeClassifier(
        max_depth=4,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
    )
    tree_tuned.fit(X_train, y_train)

    # Train a Random Forest with multiple decision trees
    forest = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42,
    )
    forest.fit(X_train, y_train)

    # Evaluate each model
    models = {
        "Decision Tree (Default)": tree_default,
        "Decision Tree (Tuned)": tree_tuned,
        "Random Forest": forest,
    }

    results = []

    for name, model in models.items():
        train_accuracy = accuracy_score(
            y_train, model.predict(X_train)
        )
        predictions = model.predict(X_test)

        results.append({
            "Model": name,
            "Train Accuracy": train_accuracy,
            "Test Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions),
            "Recall": recall_score(y_test, predictions),
            "F1 Score": f1_score(y_test, predictions),
        })

    results_df = pd.DataFrame(results)

    print("\nMODEL COMPARISON")
    print(results_df.round(4).to_string(index=False))

    # Save the model comparison table
    results_df.to_csv(
        os.path.join(output_dir, "model_comparison.csv"),
        index=False,
    )

    # Visualise the tuned Decision Tree
    plt.figure(figsize=(22, 10))
    plot_tree(
        tree_tuned,
        feature_names=X.columns,
        class_names=list(dataset.target_names),
        filled=True,
        rounded=True,
        fontsize=7,
    )
    plt.title("Tuned Decision Tree")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "tuned_decision_tree.png"),
        dpi=200,
    )
    plt.close()

    # Plot feature importance from the Random Forest
    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": forest.feature_importances_,
    }).sort_values("Importance", ascending=False)

    importance.to_csv(
        os.path.join(output_dir, "random_forest_feature_importance.csv"),
        index=False,
    )

    top_features = importance.head(10).sort_values("Importance")

    plt.figure(figsize=(10, 6))
    plt.barh(top_features["Feature"], top_features["Importance"])
    plt.xlabel("Feature Importance")
    plt.title("Top 10 Random Forest Features")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "random_forest_feature_importance.png"),
        dpi=200,
    )
    plt.close()

    # Save a confusion matrix for the Random Forest
    forest_predictions = forest.predict(X_test)
    matrix = confusion_matrix(y_test, forest_predictions)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=dataset.target_names,
    )
    display.plot(cmap="Blues")
    plt.title("Random Forest Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "random_forest_confusion_matrix.png"),
        dpi=200,
    )
    plt.close()

    print("\nSaved Day 3 outputs to:", output_dir)
    print("Files created:")
    print("- model_comparison.csv")
    print("- tuned_decision_tree.png")
    print("- random_forest_feature_importance.csv")
    print("- random_forest_feature_importance.png")
    print("- random_forest_confusion_matrix.png")
    print("\nDay 3 completed successfully.")

day = 3

if day == 1:
    run_day1()
elif day == 2:
    run_day2()
elif day == 3:
    run_day3()
    

def run_day4():
    """Train and compare SVM and KNN classification models."""

    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        ConfusionMatrixDisplay,
    )

    # Create a separate folder for Day 4 results
    output_dir = "week3_day4_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Load the dataset
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target

    print("\nWEEK 3 - DAY 4: SVM & KNN")
    print("-" * 45)
    print("Dataset shape:", X.shape)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Standardise features because SVM and KNN are scale-sensitive
    svm_model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1.0),
    )

    knn_model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5),
    )

    # Train both models
    models = {
        "SVM": svm_model,
        "KNN": knn_model,
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions),
            "Recall": recall_score(y_test, predictions),
            "F1 Score": f1_score(y_test, predictions),
        })

        # Save a confusion matrix for each model
        display = ConfusionMatrixDisplay.from_estimator(
            model,
            X_test,
            y_test,
            display_labels=dataset.target_names,
            cmap="Blues",
        )
        plt.title(f"{name} Confusion Matrix")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{name.lower()}_confusion_matrix.png"),
            dpi=200,
        )
        plt.close()

    # Display and save the comparison table
    results_df = pd.DataFrame(results)

    print("\nMODEL COMPARISON")
    print(results_df.round(4).to_string(index=False))

    results_df.to_csv(
        os.path.join(output_dir, "model_comparison.csv"),
        index=False,
    )

    # Plot a comparison of model accuracy
    plt.figure(figsize=(7, 5))
    plt.bar(results_df["Model"], results_df["Accuracy"])
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title("SVM vs KNN Accuracy")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "accuracy_comparison.png"),
        dpi=200,
    )
    plt.close()

    print("\nNote: SVM and KNN do not provide Random Forest-style")
    print("feature_importances_. This task compares their predictions.")
    print("\nDay 4 results saved in:", output_dir)
    print("Files created:")
    print("- model_comparison.csv")
    print("- svm_confusion_matrix.png")
    print("- knn_confusion_matrix.png")
    print("- accuracy_comparison.png")
    print("\nDay 4 completed successfully.")

day = 4

if day == 1:
    run_day1()
elif day == 2:
    run_day2()
elif day == 3:
    run_day3()
elif day == 4:
    run_day4()


def run_day5():
    """Tune SVM and KNN using GridSearchCV and RandomizedSearchCV."""

    import os
    import pandas as pd

    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import (
        train_test_split,
        GridSearchCV,
        RandomizedSearchCV,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from scipy.stats import randint

    # Create a separate folder for Day 5 results
    output_dir = "week3_day5_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Load the dataset
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target

    print("\nWEEK 3 - DAY 5: HYPERPARAMETER TUNING")
    print("-" * 50)
    print("Dataset shape:", X.shape)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Pipeline scales features before training the model
    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC()),
    ])

    knn_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier()),
    ])

    # Parameter combinations for Grid Search
    svm_grid = {
        "model__C": [0.1, 1, 10],
        "model__kernel": ["linear", "rbf"],
        "model__gamma": ["scale", "auto"],
    }

    knn_grid = {
        "model__n_neighbors": [3, 5, 7, 9],
        "model__weights": ["uniform", "distance"],
        "model__metric": ["euclidean", "manhattan"],
    }

    # Run GridSearchCV for SVM
    print("\nRunning Grid Search for SVM...")
    svm_search = GridSearchCV(
        svm_pipeline,
        svm_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    svm_search.fit(X_train, y_train)

    # Run GridSearchCV for KNN
    print("Running Grid Search for KNN...")
    knn_search = GridSearchCV(
        knn_pipeline,
        knn_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    knn_search.fit(X_train, y_train)

    # Randomized search parameter distributions
    svm_random_params = {
        "model__C": [0.01, 0.1, 1, 10, 100],
        "model__kernel": ["linear", "rbf"],
        "model__gamma": ["scale", "auto", 0.001, 0.01, 0.1],
    }

    knn_random_params = {
        "model__n_neighbors": randint(3, 16),
        "model__weights": ["uniform", "distance"],
        "model__metric": ["euclidean", "manhattan"],
    }

    # Run RandomizedSearchCV for SVM
    print("\nRunning Randomized Search for SVM...")
    svm_random_search = RandomizedSearchCV(
        svm_pipeline,
        svm_random_params,
        n_iter=10,
        cv=5,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1,
    )
    svm_random_search.fit(X_train, y_train)

    # Run RandomizedSearchCV for KNN
    print("Running Randomized Search for KNN...")
    knn_random_search = RandomizedSearchCV(
        knn_pipeline,
        knn_random_params,
        n_iter=10,
        cv=5,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1,
    )
    knn_random_search.fit(X_train, y_train)

    # Evaluate each best model on the test set
    searches = {
        "SVM - Grid Search": svm_search,
        "KNN - Grid Search": knn_search,
        "SVM - Random Search": svm_random_search,
        "KNN - Random Search": knn_random_search,
    }

    results = []

    for name, search in searches.items():
        predictions = search.best_estimator_.predict(X_test)

        results.append({
            "Search": name,
            "Best CV Accuracy": search.best_score_,
            "Test Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions),
            "Recall": recall_score(y_test, predictions),
            "F1 Score": f1_score(y_test, predictions),
            "Best Parameters": str(search.best_params_),
        })

        print(f"\n{name}")
        print("Best parameters:", search.best_params_)
        print("Best cross-validation accuracy:", round(search.best_score_, 4))

    # Display and save the comparison table
    results_df = pd.DataFrame(results)

    print("\nFINAL MODEL COMPARISON")
    print(results_df.round(4).to_string(index=False))

    results_df.to_csv(
        os.path.join(output_dir, "hyperparameter_tuning_results.csv"),
        index=False,
    )

    # Save the full Grid Search results
    pd.DataFrame(svm_search.cv_results_).to_csv(
        os.path.join(output_dir, "svm_grid_search_details.csv"),
        index=False,
    )

    pd.DataFrame(knn_search.cv_results_).to_csv(
        os.path.join(output_dir, "knn_grid_search_details.csv"),
        index=False,
    )

    print("\nDay 5 results saved in:", output_dir)
    print("Files created:")
    print("- hyperparameter_tuning_results.csv")
    print("- svm_grid_search_details.csv")
    print("- knn_grid_search_details.csv")
    print("\nDay 5 completed successfully.")

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


