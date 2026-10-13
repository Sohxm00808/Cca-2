"""
Name : Jaylaksh
PRN  : 1302250524
Topic: Traditional Programming vs Machine Learning (Loan Approval)

Traditional programming : humans write the RULES, the program applies them to data.
Machine learning        : we give DATA + ANSWERS (Training sheet), the model learns the rules.

Input  : input_data.xlsx  -> sheets 'Training' (with Approved) and 'Input' (new applicants)
Output : written back to the same file, sheet 'Output'
"""
import argparse
import json
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

FEATURES = ["MonthlyIncome", "CreditScore", "ExistingEMI", "LoanAmount"]


def traditional_decision(row):
    if row["CreditScore"] < 700:
        return "No"
    if row["ExistingEMI"] > 0.40 * row["MonthlyIncome"]:
        return "No"
    if row["LoanAmount"] > 20 * row["MonthlyIncome"]:
        return "No"
    return "Yes"


def train_model(train_df, max_depth=4, random_state=42):
    X, y = train_df[FEATURES], train_df["Status"] if "Status" in train_df.columns else train_df["Approved"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=random_state)
    
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state).fit(X_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te))
    model.fit(X, y)  # Refit on all training data
    return model, acc


def run_evaluation(file="input_data.xlsx", max_depth=4, output_json="evaluation_summary.json"):
    print(f"Reading dataset from '{file}'...")
    
    # Check sheet names dynamically
    xls = pd.ExcelFile(file)
    train_sheet = "Training Data" if "Training Data" in xls.sheet_names else "Training"
    input_sheet = "Inference" if "Inference" in xls.sheet_names else "Input"

    train_df = pd.read_excel(file, sheet_name=train_sheet)
    input_df = pd.read_excel(file, sheet_name=input_sheet)

    out = input_df.copy()
    out["Traditional_Decision"] = input_df.apply(traditional_decision, axis=1)

    model, acc = train_model(train_df, max_depth=max_depth)
    out["ML_Decision"] = model.predict(input_df[FEATURES])
    
    # Add probability confidence scores for extra insight
    probs = model.predict_proba(input_df[FEATURES])
    classes = list(model.classes_)
    yes_idx = classes.index("Yes") if "Yes" in classes else 1
    out["ML_Confidence"] = [round(p[yes_idx], 2) for p in probs]

    out["Same_Result"] = (out["Traditional_Decision"] == out["ML_Decision"]).map({True: "Yes", False: "No"})

    # Write back to Excel
    with pd.ExcelWriter(file, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
        out.to_excel(w, sheet_name="Comparison_Output", index=False)

    agreed_count = (out["Same_Result"] == "Yes").sum()
    total_rows = len(out)
    
    print(f"\n--- Evaluation Results ---")
    print(f"ML Test Accuracy: {acc:.2%}")
    print(f"Traditional vs ML agreement: {agreed_count}/{total_rows} applicants")
    print(out.to_string(index=False))

    # Export summary report
    summary = {
        "file": file,
        "test_accuracy": round(acc, 4),
        "total_evaluated": total_rows,
        "agreement_count": int(agreed_count),
        "agreement_percentage": round(100 * agreed_count / total_rows, 2)
    }
    with open(output_json, "w", encoding="utf-8") as jf:
        json.dump(summary, jf, indent=4)
    print(f"\nSuccessfully exported summary to '{output_json}' and updated '{file}' (Sheet: Comparison_Output).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate traditional rules vs. Decision Tree model on loan data.")
    parser.add_argument("--file", type=str, default="input_data.xlsx", help="Target Excel file path")
    parser.add_argument("--depth", type=int, default=4, help="Max depth for DecisionTreeClassifier")
    parser.add_argument("--json", type=str, default="evaluation_summary.json", help="Path for JSON summary output")
    args = parser.parse_args()

    run_evaluation(file=args.file, max_depth=args.depth, output_json=args.json)
