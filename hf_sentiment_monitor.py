"""
Name : abhishek raskar 
PRN  : 1302250009
Topic: ML Framework - Hugging Face Sentiment Monitor (multiple problem statements)

Uses a Hugging Face pipeline when `transformers` is installed (pip install transformers torch).
If it is not available, a small keyword fallback runs so the script still works offline.
"""
import random
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill

COLS = ["ApplicantID", "MonthlyIncome", "CreditScore", "ExistingEMI", "LoanAmount"]


def make_row(i):
    # Slightly varied income and credit score distributions
    inc = random.randrange(25000, 160000, 5000)
    cs = random.randrange(580, 840, 10)
    emi = int(inc * random.uniform(0.1, 0.65) // 500 * 500)
    loan = int(inc * random.uniform(4, 35) // 10000 * 10000)
    return [i, inc, cs, emi, loan]


def historical_label(r):
    # Slightly adjusted scoring weights for collaborator's branch
    _, inc, cs, emi, loan = r
    score = (cs - 620) / 220 - emi / inc * 0.85 - loan / (inc * 55)
    return "Yes" if score > 0.08 else "No"


def create(file="input_data_v2.xlsx"):
    random.seed(42)  # Different random seed for collaborator
    wb = Workbook()
    
    train = wb.active
    train.title = "Training"
    train.append(COLS + ["Approved"])
    for i in range(1, 61):
        r = make_row(i)
        train.append(r + [historical_label(r)])

    new = wb.create_sheet("Input")
    new.append(COLS)
    for i in range(101, 116):
        new.append(make_row(i))

    for sheet in (train, new):
        for c in sheet[1]:
            c.font = Font(bold=True, color="FFFFFF")
            # Changed header fill color to Dark Slate / Charcoal for variation
            c.fill = PatternFill("solid", fgColor="333333")
        for col in "ABCDEF":
            sheet.column_dimensions[col].width = 18

    wb.save(file)
    print(file, "created successfully for collaborator branch.")


if __name__ == "__main__":
    create()
