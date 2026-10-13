def run_day1():
    """Evaluate a classification model using metrics, cross-validation,
    and learning curves.
    """

    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import (
        train_test_split,
        cross_validate,
        StratifiedKFold,
        KFold,
        learning_curve,
    )
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        classification_report,
        ConfusionMatrixDisplay,
    )

    # Create a separate folder for Day 1 results
    output_dir = "week4_day1_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Load the classification dataset
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target

    print("\nWEEK 4 - DAY 1: MODEL EVALUATION METRICS")
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

    # Scale features and train Logistic Regression
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, random_state=42),
    )
    model.fit(X_train, y_train)

    # Evaluate the model on the test set
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1 Score": f1_score(y_test, predictions),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
    }

    print("\nTEST SET METRICS")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nCLASSIFICATION REPORT")
    print(classification_report(
        y_test,
        predictions,
        target_names=dataset.target_names,
    ))

    # Save evaluation metrics
    pd.DataFrame(
        list(metrics.items()),
        columns=["Metric", "Score"],
    ).to_csv(
        os.path.join(output_dir, "evaluation_metrics.csv"),
        index=False,
    )

    # Save the confusion matrix
    display = ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        display_labels=dataset.target_names,
        cmap="Blues",
    )
    plt.title("Logistic Regression - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "confusion_matrix.png"),
        dpi=200,
    )
    plt.close()

    # Compare ordinary K-Fold and Stratified K-Fold CV
    cv_methods = {
        "K-Fold": KFold(n_splits=5, shuffle=True, random_state=42),
        "Stratified K-Fold": StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42,
        ),
    }

    cv_results = []

    for name, cv in cv_methods.items():
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=["accuracy", "precision", "recall", "roc_auc"],
            n_jobs=-1,
        )

        cv_results.append({
            "Method": name,
            "Mean Accuracy": scores["test_accuracy"].mean(),
            "Mean Precision": scores["test_precision"].mean(),
            "Mean Recall": scores["test_recall"].mean(),
            "Mean ROC-AUC": scores["test_roc_auc"].mean(),
        })

    cv_df = pd.DataFrame(cv_results)

    print("\nCROSS-VALIDATION COMPARISON")
    print(cv_df.round(4).to_string(index=False))

    cv_df.to_csv(
        os.path.join(output_dir, "cross_validation_comparison.csv"),
        index=False,
    )

    # Generate learning curves to diagnose overfitting and underfitting
    sizes, train_scores, validation_scores = learning_curve(
        model,
        X_train,
        y_train,
        cv=StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42,
        ),
        scoring="accuracy",
        train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
        n_jobs=-1,
    )

    train_means = train_scores.mean(axis=1)
    validation_means = validation_scores.mean(axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(sizes, train_means, marker="o", label="Training accuracy")
    plt.plot(
        sizes,
        validation_means,
        marker="o",
        label="Validation accuracy",
    )
    plt.xlabel("Number of Training Samples")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve - Logistic Regression")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "learning_curve.png"),
        dpi=200,
    )
    plt.close()

    # Save the learning curve values
    pd.DataFrame({
        "Training Samples": sizes,
        "Training Accuracy": train_means,
        "Validation Accuracy": validation_means,
    }).to_csv(
        os.path.join(output_dir, "learning_curve_data.csv"),
        index=False,
    )

    print("\nDay 1 results saved in:", output_dir)
    print("Files created:")
    print("- evaluation_metrics.csv")
    print("- confusion_matrix.png")
    print("- cross_validation_comparison.csv")
    print("- learning_curve.png")
    print("- learning_curve_data.csv")
    print("\nDay 1 completed successfully.")


# Select which day's task to run
day = 1

if day == 1:
    run_day1()


def run_day2():
    """Study bias-variance tradeoff, regularisation, and hyperparameter tuning."""

    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import (
        train_test_split,
        GridSearchCV,
        RandomizedSearchCV,
        learning_curve,
        StratifiedKFold,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from scipy.stats import loguniform

    # Create a separate folder for Day 2 results
    output_dir = "week4_day2_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Load the dataset
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target

    print("\nWEEK 4 - DAY 2: BIAS-VARIANCE & REGULARISATION")
    print("-" * 55)
    print("Dataset shape:", X.shape)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Standardise features before Logistic Regression
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000, random_state=42)),
    ])

    # Compare weakly and strongly regularised models
    regularisation_results = []

    for c_value in [0.001, 0.01, 0.1, 1, 10, 100]:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                C=c_value,
                max_iter=5000,
                random_state=42,
            )),
        ])

        model.fit(X_train, y_train)

        regularisation_results.append({
            "C": c_value,
            "Training Accuracy": accuracy_score(
                y_train, model.predict(X_train)
            ),
            "Test Accuracy": accuracy_score(
                y_test, model.predict(X_test)
            ),
        })

    regularisation_df = pd.DataFrame(regularisation_results)

    print("\nREGULARISATION COMPARISON")
    print(regularisation_df.round(4).to_string(index=False))

    regularisation_df.to_csv(
        os.path.join(output_dir, "regularisation_comparison.csv"),
        index=False,
    )

    # Grid Search: test a defined set of hyperparameters
    grid_parameters = {
        "model__C": [0.01, 0.1, 1, 10, 100],
        "model__solver": ["liblinear", "lbfgs"],
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    print("\nRunning Grid Search...")
    grid_search = GridSearchCV(
        pipeline,
        grid_parameters,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    # Randomized Search: sample hyperparameters from distributions
    random_parameters = {
        "model__C": loguniform(0.001, 100),
        "model__solver": ["liblinear", "lbfgs"],
    }

    print("Running Randomized Search...")
    random_search = RandomizedSearchCV(
        pipeline,
        random_parameters,
        n_iter=12,
        cv=cv,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1,
    )
    random_search.fit(X_train, y_train)

    # Evaluate the best models on the held-out test set
    search_results = []

    for name, search in [
        ("Grid Search", grid_search),
        ("Randomized Search", random_search),
    ]:
        best_model = search.best_estimator_

        search_results.append({
            "Search Method": name,
            "Best CV Accuracy": search.best_score_,
            "Test Accuracy": accuracy_score(
                y_test, best_model.predict(X_test)
            ),
            "Best Parameters": str(search.best_params_),
        })

        print(f"\n{name}")
        print("Best parameters:", search.best_params_)
        print("Best CV accuracy:", round(search.best_score_, 4))

    search_df = pd.DataFrame(search_results)

    print("\nGRID SEARCH VS RANDOMIZED SEARCH")
    print(search_df.round(4).to_string(index=False))

    search_df.to_csv(
        os.path.join(output_dir, "search_comparison.csv"),
        index=False,
    )

    # Plot regularisation results to compare training and test accuracy
    plt.figure(figsize=(8, 5))
    plt.semilogx(
        regularisation_df["C"],
        regularisation_df["Training Accuracy"],
        marker="o",
        label="Training Accuracy",
    )
    plt.semilogx(
        regularisation_df["C"],
        regularisation_df["Test Accuracy"],
        marker="o",
        label="Test Accuracy",
    )
    plt.xlabel("C (Inverse Regularisation Strength)")
    plt.ylabel("Accuracy")
    plt.title("Effect of Regularisation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "regularisation_comparison.png"),
        dpi=200,
    )
    plt.close()

    # Learning curve to compare training and validation performance
    sizes, train_scores, validation_scores = learning_curve(
        grid_search.best_estimator_,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
        train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
        n_jobs=-1,
    )

    plt.figure(figsize=(8, 5))
    plt.plot(sizes, train_scores.mean(axis=1), marker="o",
             label="Training Accuracy")
    plt.plot(sizes, validation_scores.mean(axis=1), marker="o",
             label="Validation Accuracy")
    plt.xlabel("Number of Training Samples")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve - Tuned Logistic Regression")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "learning_curve.png"),
        dpi=200,
    )
    plt.close()

    print("\nDay 2 results saved in:", output_dir)
    print("Files created:")
    print("- regularisation_comparison.csv")
    print("- search_comparison.csv")
    print("- regularisation_comparison.png")
    print("- learning_curve.png")
    print("\nDay 2 completed successfully.")

    
day = 2

if day == 1:
    run_day1()
elif day == 2:
    run_day2()


def run_day3():
    import os
    import pickle
    import joblib
    import numpy as np
    import pandas as pd

    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, roc_auc_score
    )

    # Create a separate output folder
    output_dir = "week4_day3_outputs"
    os.makedirs(output_dir, exist_ok=True)

    print("\n--- Week 4 Day 3: Model Serialisation ---")

    # Load the dataset
    dataset = load_breast_cancer()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(dataset.target)

    # Split the data reproducibly
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build and train the model pipeline
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000))
    ])
    model.fit(X_train, y_train)

    # Evaluate the original model
    original_predictions = model.predict(X_test)
    original_probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, original_predictions),
        "precision": precision_score(y_test, original_predictions),
        "recall": recall_score(y_test, original_predictions),
        "f1_score": f1_score(y_test, original_predictions),
        "roc_auc": roc_auc_score(y_test, original_probabilities)
    }

    pd.DataFrame([metrics]).to_csv(
        os.path.join(output_dir, "model_metrics.csv"), index=False
    )

    # Save using joblib
    joblib_path = os.path.join(output_dir, "model_joblib.pkl")
    joblib.dump(model, joblib_path)
    loaded_joblib_model = joblib.load(joblib_path)

    # Save using pickle
    pickle_path = os.path.join(output_dir, "model_pickle.pkl")
    with open(pickle_path, "wb") as file:
        pickle.dump(model, file)

    with open(pickle_path, "rb") as file:
        loaded_pickle_model = pickle.load(file)

    # Compare predictions after loading
    joblib_predictions = loaded_joblib_model.predict(X_test)
    pickle_predictions = loaded_pickle_model.predict(X_test)

    results = {
        "Original vs joblib predictions match": np.array_equal(
            original_predictions, joblib_predictions
        ),
        "Original vs pickle predictions match": np.array_equal(
            original_predictions, pickle_predictions
        ),
        "Test samples": len(y_test)
    }

    pd.DataFrame([results]).to_csv(
        os.path.join(output_dir, "serialization_test.csv"), index=False
    )

    print("\nEvaluation metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nSerialization verification:")
    for name, value in results.items():
        print(f"{name}: {value}")

    print(f"\nFiles saved in: {output_dir}")
    print("Day 3 completed successfully.")

day = 3

if day == 1:
    run_day1()
elif day == 2:
    run_day2()
elif day == 3:
    run_day3()


def run_day4():
    import os
    import subprocess
    import sys
    import time
    import socket
    import pickle
    from pathlib import Path

    import joblib
    import numpy as np
    import pandas as pd
    from fastapi import FastAPI
    from pydantic import BaseModel
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    output_dir = "week4_day4_outputs"
    os.makedirs(output_dir, exist_ok=True)

    print("\n--- Week 4 Day 4: FastAPI Model Serving ---")

    # Train the model using the same dataset and feature order
    dataset = load_breast_cancer()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(dataset.target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000))
    ])
    model.fit(X_train, y_train)

    # Save the trained model and required feature names
    model_path = os.path.abspath(
        os.path.join(output_dir, "api_model.joblib")
    )
    joblib.dump(model, model_path)

    feature_names_path = os.path.join(output_dir, "feature_names.txt")
    with open(feature_names_path, "w", encoding="utf-8") as file:
        file.write("\n".join(dataset.feature_names))

    # Generate a standalone FastAPI application
    api_file = os.path.abspath(
        os.path.join(output_dir, "model_api.py")
    )

    api_code = '''
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "api_model.joblib")
FEATURE_PATH = os.path.join(BASE_DIR, "feature_names.txt")

model = joblib.load(MODEL_PATH)

with open(FEATURE_PATH, encoding="utf-8") as file:
    FEATURE_NAMES = [line.strip() for line in file if line.strip()]

app = FastAPI(title="Breast Cancer Prediction API")

class PredictionInput(BaseModel):
    features: dict[str, float] = Field(
        ..., description="Feature names and their numeric values"
    )

@app.get("/")
def home():
    return {"message": "Model API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: PredictionInput):
    if set(data.features.keys()) != set(FEATURE_NAMES):
        raise HTTPException(
            status_code=400,
            detail="Provide all required feature names, with no extras."
        )

    try:
        row = pd.DataFrame(
            [[data.features[name] for name in FEATURE_NAMES]],
            columns=FEATURE_NAMES
        )
        prediction = int(model.predict(row)[0])
        probabilities = model.predict_proba(row)[0]

        return {
            "prediction": prediction,
            "predicted_class": (
                "malignant" if prediction == 0 else "benign"
            ),
            "probabilities": {
                "malignant": float(probabilities[0]),
                "benign": float(probabilities[1])
            }
        }
    except (ValueError, TypeError) as error:
        raise HTTPException(status_code=400, detail=str(error))
'''

    with open(api_file, "w", encoding="utf-8") as file:
        file.write(api_code)

    # Test the trained model before starting the API
    accuracy = accuracy_score(y_test, model.predict(X_test))
    pd.DataFrame([{"test_accuracy": accuracy}]).to_csv(
        os.path.join(output_dir, "model_metrics.csv"), index=False
    )

    # Find an available local port
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    # Start FastAPI as a separate process
    server = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "model_api:app",
            "--host", "127.0.0.1",
            "--port", str(port)
        ],
        cwd=output_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    try:
        import httpx

        base_url = f"http://127.0.0.1:{port}"
        response = None

        # Wait for the server to start
        for _ in range(30):
            if server.poll() is not None:
                raise RuntimeError(
                    "FastAPI server stopped. Check FastAPI/Uvicorn installation."
                )
            try:
                response = httpx.get(
                    f"{base_url}/health", timeout=1.0
                )
                if response.status_code == 200:
                    break
            except httpx.RequestError:
                pass
            time.sleep(1)

        if response is None or response.status_code != 200:
            raise RuntimeError("FastAPI server did not start in time.")

        # Test the prediction endpoint with one test sample
        sample = X_test.iloc[0].to_dict()
        prediction_response = httpx.post(
            f"{base_url}/predict",
            json={"features": sample},
            timeout=10.0
        )
        prediction_response.raise_for_status()

        result = prediction_response.json()
        with open(
            os.path.join(output_dir, "api_test_result.json"),
            "w",
            encoding="utf-8"
        ) as file:
            import json
            json.dump(result, file, indent=4)

        print(f"Test accuracy: {accuracy:.4f}")
        print(f"Health endpoint: {response.status_code}")
        print(f"Prediction endpoint: {prediction_response.status_code}")
        print(f"Prediction result: {result}")
        print(f"Files saved in: {output_dir}")
        print("Day 4 completed successfully.")

    finally:
        # Stop the temporary server after testing
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()

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
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        classification_report,
        accuracy_score,
        precision_score,
        recall_score,
        confusion_matrix,
        ConfusionMatrixDisplay,
        roc_curve,
        roc_auc_score
    )

    # Create a separate output folder
    output_dir = "week4_day5_outputs"
    os.makedirs(output_dir, exist_ok=True)

    print("\n--- Week 4 Day 5: Classification Capstone ---")

    # Load a binary classification dataset
    dataset = load_breast_cancer()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(dataset.target)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Logistic Regression
    logistic_model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000))
    ])
    logistic_model.fit(X_train, y_train)

    logistic_predictions = logistic_model.predict(X_test)
    logistic_probabilities = logistic_model.predict_proba(X_test)[:, 1]

    print("\nClassification Report — Logistic Regression")
    report = classification_report(
        y_test,
        logistic_predictions,
        target_names=dataset.target_names,
        zero_division=0
    )
    print(report)

    with open(
        os.path.join(output_dir, "classification_report.txt"),
        "w",
        encoding="utf-8"
    ) as file:
        file.write(report)

    # Plot and save the confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        logistic_predictions,
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

    # Plot and save the ROC curve
    fpr, tpr, _ = roc_curve(y_test, logistic_probabilities)
    logistic_auc = roc_auc_score(y_test, logistic_probabilities)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"Logistic Regression (AUC={logistic_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Logistic Regression - ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "roc_auc_curve.png"),
        dpi=150
    )
    plt.close()

    # Train Random Forest
    forest_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )
    forest_model.fit(X_train, y_train)
    forest_predictions = forest_model.predict(X_test)

    # Compare both models using the same test data
    comparison = pd.DataFrame([
        {
            "Model": "Logistic Regression",
            "Accuracy": accuracy_score(y_test, logistic_predictions),
            "Precision": precision_score(
                y_test, logistic_predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, logistic_predictions, zero_division=0
            ),
            "ROC_AUC": logistic_auc
        },
        {
            "Model": "Random Forest",
            "Accuracy": accuracy_score(y_test, forest_predictions),
            "Precision": precision_score(
                y_test, forest_predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, forest_predictions, zero_division=0
            ),
            "ROC_AUC": roc_auc_score(
                y_test, forest_model.predict_proba(X_test)[:, 1]
            )
        }
    ])

    comparison.to_csv(
        os.path.join(output_dir, "model_comparison.csv"),
        index=False
    )

    print("\nModel Comparison")
    print(comparison.round(4).to_string(index=False))

    # Plot model comparison
    comparison.set_index("Model")[
        ["Accuracy", "Precision", "Recall", "ROC_AUC"]
    ].plot(kind="bar", figsize=(9, 5), ylim=(0, 1))
    plt.title("Logistic Regression vs Random Forest")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "model_comparison.png"),
        dpi=150
    )
    plt.close()

    print(f"\nFiles saved in: {output_dir}")
    print("Day 5 completed successfully.")

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

