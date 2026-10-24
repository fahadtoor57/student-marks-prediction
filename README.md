# Student Marks Prediction System

A beginner-friendly machine learning project that predicts a student's final exam marks from study hours, attendance, and previous marks.

## Technologies
- Python
- Pandas
- NumPy
- Scikit-learn
- SQLite (`sqlite3`, included with Python)

## Algorithm
- Linear Regression

## Project files
- `students.csv` — small synthetic learning dataset
- `student_marks.py` — cleans data, trains/evaluates the model, predicts marks, and saves data/results to SQLite
- `database.py` — demonstrates SQL queries
- `requirements.txt` — Python package dependencies
- `students.db` — created automatically when the main script runs

## Setup (Windows / VS Code)

1. Install Python.
2. Open this folder in VS Code.
3. Open Terminal → New Terminal.
4. (Recommended) Create a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

5. Install packages:

   ```powershell
   python -m pip install -r requirements.txt
   ```

6. Run the project:

   ```powershell
   python student_marks.py
   ```

7. Enter study hours, attendance percentage, and previous marks when prompted.
8. View the SQL queries and saved results:

   ```powershell
   python database.py
   ```

## Dataset columns
- `StudyHours`: daily study hours
- `Attendance`: percentage from 0 to 100
- `PreviousMarks`: previous exam marks from 0 to 100
- `FinalMarks`: target marks used for training/evaluation

The included 20 records are synthetic examples for learning, not real student data.

## Model evaluation
The script prints:
- MAE (mean absolute error)
- MSE (mean squared error)
- RMSE (root mean squared error)
- R² score

Because this is a very small dataset, test metrics are only a demonstration and are not evidence of real-world predictive performance.

## SQLite tables
After running `student_marks.py`, `students.db` includes:
- `students`
- `predictions`
- `new_student_predictions` (created after a successful interactive prediction)

## Limitations and next steps
- Gather a larger, representative dataset before making real predictions.
- Compare against a simple baseline and evaluate on more data.
- Do not use predictions as the sole basis for important decisions about students.
- Keep the test set separate from training when making improvements.
