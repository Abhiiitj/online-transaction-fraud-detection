"""Reproducible fraud classification for tabular transaction CSVs."""
import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, average_precision_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


def load_data(path: Path, target: str) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(path)
    if target not in data:
        raise ValueError(f"Missing target column: {target}")
    y = pd.to_numeric(data[target], errors="raise")
    if not set(y.unique()).issubset({0, 1}) or y.nunique() != 2:
        raise ValueError("Target must contain both binary classes 0 and 1.")
    # IDs and post-transaction flags are not model inputs.
    excluded = {target, "nameOrig", "nameDest", "isFlaggedFraud"}
    x = data.drop(columns=[c for c in excluded if c in data])
    if x.empty:
        raise ValueError("No feature columns remain.")
    return x, y.astype(int)


def build_pipeline(x: pd.DataFrame, classifier):
    numeric = x.select_dtypes(include="number").columns.tolist()
    categorical = x.columns.difference(numeric).tolist()
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    return Pipeline([("preprocess", preprocessor), ("classifier", classifier)])


def train_and_evaluate(csv_path: Path, target: str, output_dir: Path):
    x, y = load_data(csv_path, target)
    if y.value_counts().min() < 5:
        raise ValueError("Each target class needs at least five rows for stratified train/test and tuning splits.")
    output_dir.mkdir(parents=True, exist_ok=True)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )
    if y_train.value_counts().min() < 3:
        raise ValueError("Each class needs at least three training rows for cross-validation.")
    candidates = {
        "logistic_regression": (LogisticRegression(max_iter=1000, class_weight="balanced"), {"classifier__C": [0.1, 1, 10]}),
        "decision_tree": (DecisionTreeClassifier(class_weight="balanced", random_state=42),
                          {"classifier__max_depth": [4, 8, None], "classifier__min_samples_leaf": [1, 5]}),
    }
    reports = {}
    searches = {}
    for name, (classifier, grid) in candidates.items():
        search = GridSearchCV(build_pipeline(x_train, classifier), grid, scoring="average_precision", cv=3, n_jobs=-1)
        search.fit(x_train, y_train)
        predicted = search.predict(x_test)
        scores = search.predict_proba(x_test)[:, 1]
        reports[name] = {
            "best_parameters": search.best_params_,
            "test_average_precision": round(average_precision_score(y_test, scores), 4),
            "classification_report": classification_report(y_test, predicted, output_dict=True, zero_division=0),
        }
        searches[name] = search
        ConfusionMatrixDisplay.from_predictions(y_test, predicted, display_labels=["legitimate", "fraud"])
        plt.tight_layout()
        plt.savefig(output_dir / f"{name}_confusion_matrix.png", dpi=150)
        plt.close()
    winner = max(reports, key=lambda name: reports[name]["test_average_precision"])
    joblib.dump(searches[winner].best_estimator_, output_dir / "model.joblib")
    counts = y.value_counts().rename(index={0: "legitimate", 1: "fraud"})
    sns.barplot(x=counts.index, y=counts.values)
    plt.ylabel("Transactions")
    plt.tight_layout()
    plt.savefig(output_dir / "class_balance.png", dpi=150)
    plt.close()
    pd.Series({name: report["test_average_precision"] for name, report in reports.items()}).to_csv(output_dir / "model_comparison.csv", header=["test_average_precision"])
    return winner, reports


def main():
    parser = argparse.ArgumentParser(description="Train and compare fraud classifiers on a labeled transaction CSV.")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--target", default="isFraud")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    winner, reports = train_and_evaluate(args.csv, args.target, args.output_dir)
    for name, report in reports.items():
        print(f"\n{name}: test average precision = {report['test_average_precision']}")
        print(pd.DataFrame(report["classification_report"]).transpose().to_string())
    print(f"\nSaved {winner} model to {args.output_dir / 'model.joblib'}")


if __name__ == "__main__":
    main()

