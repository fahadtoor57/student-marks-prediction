"""
Student Marks Prediction System
Technologies: Python, Pandas, NumPy, Scikit-learn, SQLite
Algorithm: Linear Regression
"""

from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "students.csv"
DB_PATH = BASE_DIR / "students.db"


def load_and_clean_data():
    """Load the CSV, validate columns, and clean basic data issues."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    required_columns = [
        "StudyHours", "Attendance", "PreviousMarks", "FinalMarks"
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df[required_columns].copy()
    df = df.drop_duplicates()

    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # This small teaching dataset is complete. For real data, fit imputation
    # using training data only to avoid data leakage.
    df = df.dropna(subset=required_columns)

    if df.empty:
        raise ValueError("No valid student rows remain after cleaning.")

    valid_ranges = {
        "StudyHours": df["StudyHours"].ge(0),
        "Attendance": df["Attendance"].between(0, 100),
        "PreviousMarks": df["PreviousMarks"].between(0, 100),
        "FinalMarks": df["FinalMarks"].between(0, 100),
    }
    for column, valid_mask in valid_ranges.items():
        if not valid_mask.all():
            raise ValueError(f"Invalid values found in {column}.")

    return df


def save_students_to_sql(df):
    """Store cleaned student data in SQLite."""
    with sqlite3.connect(DB_PATH) as connection:
        df.to_sql("students", connection, if_exists="replace", index=False)


def main():
    print("=" * 52)
    print("       STUDENT MARKS PREDICTION SYSTEM")
    print("=" * 52)

    df = load_and_clean_data()
    print(f"\nDataset loaded: {len(df)} student records")
    print("\nFirst five records:")
    print(df.head().to_string(index=False))

    print("\nMissing values:")
    print(df.isnull().sum().to_string())

    print("\nSummary statistics:")
    print(df.describe().round(2).to_string())

    # Save data to SQL.
    save_students_to_sql(df)
    print(f"\nStudent data saved to: {DB_PATH.name}")

    # Features and target.
    feature_columns = ["StudyHours", "Attendance", "PreviousMarks"]
    X = df[feature_columns]
    y = df["FinalMarks"]

    if len(df) < 5:
        raise ValueError("Please provide at least five valid rows to split data.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nTest-set predictions:")
    results = pd.DataFrame({
        "ActualMarks": y_test.to_numpy(),
        "PredictedMarks": np.round(y_pred, 2),
        "AbsoluteError": np.round(np.abs(y_test.to_numpy() - y_pred), 2),
    })
    print(results.to_string(index=False))

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\nModel evaluation (small sample; interpret cautiously):")
    print(f"MAE : {mae:.2f} marks")
    print(f"MSE : {mse:.2f}")
    print(f"RMSE: {rmse:.2f} marks")
    print(f"R²  : {r2:.2f}")

    # Save evaluation predictions in SQL.
    results_for_sql = results.copy()
    results_for_sql["StudentIndex"] = y_test.index.to_numpy()
    results_for_sql = results_for_sql[
        ["StudentIndex", "ActualMarks", "PredictedMarks", "AbsoluteError"]
    ]
    with sqlite3.connect(DB_PATH) as connection:
        results_for_sql.to_sql(
            "predictions", connection, if_exists="replace", index=False
        )

    print("\nSaved test predictions to the SQL table 'predictions'.")

    print("\nTry a prediction for a new student.")
    print("Enter values when prompted; press Ctrl+C to exit.")
    try:
        study_hours = float(input("Daily study hours (e.g., 6): "))
        attendance = float(input("Attendance percentage (0-100, e.g., 85): "))
        previous_marks = float(input("Previous marks (0-100, e.g., 70): "))

        if study_hours < 0:
            raise ValueError("Study hours cannot be negative.")
        if not 0 <= attendance <= 100:
            raise ValueError("Attendance must be between 0 and 100.")
        if not 0 <= previous_marks <= 100:
            raise ValueError("Previous marks must be between 0 and 100.")

        new_student = pd.DataFrame({
            "StudyHours": [study_hours],
            "Attendance": [attendance],
            "PreviousMarks": [previous_marks],
        })
        prediction = float(model.predict(new_student)[0])
        prediction = float(np.clip(prediction, 0, 100))
        print(f"\nPredicted final marks: {prediction:.2f}/100")

        with sqlite3.connect(DB_PATH) as connection:
            pd.DataFrame([{
                "StudyHours": study_hours,
                "Attendance": attendance,
                "PreviousMarks": previous_marks,
                "PredictedMarks": round(prediction, 2),
            }]).to_sql(
                "new_student_predictions", connection,
                if_exists="append", index=False
            )
        print("New-student prediction saved in SQL table 'new_student_predictions'.")

    except ValueError as error:
        print(f"Input error: {error}")
    except KeyboardInterrupt:
        print("\nPrediction input cancelled.")

    print("\nProject completed successfully.")


if __name__ == "__main__":
    main()
