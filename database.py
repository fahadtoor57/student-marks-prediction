"""Simple SQL queries for the Student Marks Prediction System."""

from pathlib import Path
import sqlite3
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "students.db"


def main():
    if not DB_PATH.exists():
        print("Database not found. Run student_marks.py first.")
        return

    with sqlite3.connect(DB_PATH) as connection:
        print("\nAll students:")
        students = pd.read_sql_query("SELECT * FROM students", connection)
        print(students.to_string(index=False))

        print("\nStudents with attendance >= 75%:")
        high_attendance = pd.read_sql_query(
            """
            SELECT StudyHours, Attendance, PreviousMarks, FinalMarks
            FROM students
            WHERE Attendance >= 75
            ORDER BY Attendance DESC
            """,
            connection,
        )
        print(high_attendance.to_string(index=False))

        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
            connection,
        )
        print("\nTables currently in the database:")
        print(tables.to_string(index=False))

        predictions = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'",
            connection,
        )
        if not predictions.empty:
            print("\nSaved test predictions:")
            result = pd.read_sql_query("SELECT * FROM predictions", connection)
            print(result.to_string(index=False))


if __name__ == "__main__":
    main()
