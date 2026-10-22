"""
Name : abhishek raskar
PRN  : 1302250009
Creates input_data.xlsx (sheets: Training, Input) used by traditional_vs_ml.py
Run  : python create_input_excel.py
"""
import random
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill

COLS = ["ApplicantID", "MonthlyIncome", "CreditScore", "ExistingEMI", "LoanAmount"]


def make_row(i):
    # Distinctly varied financial distributions for Branch 5
    inc = random.randrange(35000, 190000, 5000)
    cs = random.randrange(610, 850, 10)
    emi = int(inc * random.uniform(0.08, 0.55) // 500 * 500)
    loan = int(inc * random.uniform(3.5, 32) // 10000 * 10000)
    return [i, inc, cs, emi, loan]


def historical_label(r):
    # Unique scoring weights and threshold for Branch 5
    _, inc, cs, emi, loan = r
    score = (cs - 640) / 190 - emi / inc * 0.75 - loan / (inc * 48)
    return "Yes" if score > 0.12 else "No"


def create(file="input_data_v5.xlsx"):
    random.seed(555)  # Unique random seed for branch variant 5
    wb = Workbook()
    
    train = wb.active
    train.title = "Training"
    train.append(COLS + ["Approved"])
    
    # 80 training rows variation
    for i in range(1, 81):
        r = make_row(i)
        train.append(r + [historical_label(r)])

    new = wb.create_sheet("Input")
    new.append(COLS)
    
    # 30 input rows variation starting at ID 401
    for i in range(401, 431):
        new.append(make_row(i))

    for sheet in (train, new):
        for c in sheet[1]:
            c.font = Font(bold=True, color="FFFFFF")
            # Changed header fill color to Dark Purple for branch distinction
            c.fill = PatternFill("solid", fgColor="4A235A")
        for col in "ABCDEF":
            sheet.column_dimensions[col].width = 19

    wb.save(file)
    print(file, "created successfully for branch variant 5.")


if __name__ == "__main__":
    create()
