"""
Name : Jaylaksh
PRN  : 1302250524
Creates input_data.xlsx (sheets: Training, Input) used by traditional_vs_ml.py
Run  : python create_input_excel.py
"""
import argparse
import random
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill

COLS = ["ApplicantID", "MonthlyIncome", "CreditScore", "ExistingEMI", "LoanAmount"]


def make_row(i):
    inc = random.randrange(25000, 160000, 5000)
    cs = random.randrange(560, 840, 10)
    emi = int(inc * random.uniform(0.0, 0.65) // 500 * 500)
    loan = int(inc * random.uniform(4, 35) // 10000 * 10000)
    return [i, inc, cs, emi, loan]


def historical_label(r):
    # Modified collaborator rule weights for distinct ML model behavior
    _, inc, cs, emi, loan = r
    score = (cs - 620) / 240 - (emi / inc) * 0.85 - (loan / (inc * 55))
    return "Yes" if score > 0.03 else "No"


def create(file="loan_dataset.xlsx", seed_val=42):
    random.seed(seed_val)
    wb = Workbook()
    
    # Training Sheet
    train = wb.active
    train.title = "Training Data"
    train.append(COLS + ["Status"])
    for i in range(1, 76):  # Increased sample size to 75 rows
        r = make_row(i)
        train.append(r + [historical_label(r)])

    # Inference/Input Sheet
    new = wb.create_sheet("Inference")
    new.append(COLS)
    for i in range(201, 216):  # Different ID range for incoming batch
        new.append(make_row(i))

    # Styling updates (Collaborator accent color: #2E75B6)
    for sheet in (train, new):
        for c in sheet[1]:
            c.font = Font(bold=True, color="FFFFFF")
            c.fill = PatternFill("solid", fgColor="2E75B6")
        for col in "ABCDEF":
            sheet.column_dimensions[col].width = 18
            
    wb.save(file)
    print(f"Successfully generated collaborator dataset: {file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic loan dataset for model training.")
    parser.add_argument("--file", type=str, default="loan_dataset.xlsx", help="Output Excel filename")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    create(file=args.file, seed_val=args.seed)
