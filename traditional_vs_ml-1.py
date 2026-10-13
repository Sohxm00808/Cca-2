"""
Name :jaylaksh
PRN  : 1302250524
Topic: Traditional Programming vs Machine Learning (Loan Approval)

Traditional programming : humans write the RULES, the program applies them to data.
Machine learning        : we give DATA + ANSWERS (Training sheet), the model learns the rules.

Input  : input_data.xlsx  -> sheets 'Training' (with Approved) and 'Input' (new applicants)
Output : written back to the same file, sheet 'Output'
"""
import logging
from typing import Tuple
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Configure logging for professional output tracking
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

FILE = "input_data.xlsx"
FEATURES = ["MonthlyIncome", "CreditScore", "ExistingEMI", "LoanAmount"]


def traditional_decision(row: pd.Series) -> str:
    """Evaluate loan approval using hand-written business rules."""
    if row["CreditScore"] < 700:
        return "No"
    if row["ExistingEMI"] > 0.40 * row["MonthlyIncome"]:
        return "No"
    if row["LoanAmount"] > 20 * row["MonthlyIncome"]:
        return "No"
    return "Yes"


def train_model(train_df: pd.DataFrame) -> Tuple[DecisionTreeClassifier, float]:
    """Train a Decision Tree Classifier on historical data and evaluate test accuracy."""
    X, y = train_df[FEATURES], train_df["Approved"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=1
    )

    model = DecisionTreeClassifier(max_depth=4, random_state=1).fit(X_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te))

    # Refit on all training data for final production predictions
    model.fit(X, y)
    return model, acc


def main() -> None:
    try:
        train_df = pd.read_excel(FILE, sheet_name="Training")
        input_df = pd.read_excel(FILE, sheet_name="Input")
    except FileNotFoundError:
        logging.error(
            f"Could not find '{FILE}'. Please ensure the Excel file is placed in the root directory."
        )
        return
    except Exception as e:
        logging.error(f"Failed to read Excel sheets: {e}")
        return

    out = input_df.copy()
    out["Traditional_Decision"] = input_df.apply(traditional_decision, axis=1)

    model, acc = train_model(train_df)
    out["ML_Decision"] = model.predict(input_df[FEATURES])
    out["Same_Result"] = (
        out["Traditional_Decision"] == out["ML_Decision"]
    ).map({True: "Yes", False: "No"})

    # Write back results to the 'Output' sheet without touching other sheets
    try:
        with pd.ExcelWriter(
            FILE, engine="openpyxl", mode="a", if_sheet_exists="replace"
        ) as w:
            out.to_excel(w, sheet_name="Output", index=False)
        logging.info(f"Successfully updated '{FILE}' with the 'Output' sheet.")
    except Exception as e:
        logging.error(f"Failed to write back to Excel file: {e}")

    print(f"\nML test accuracy on held-out training data: {acc:.2%}")
    print(
        f"Traditional vs ML agree on {(out['Same_Result'] == 'Yes').sum()}/{len(out)} applicants\n"
    )
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
