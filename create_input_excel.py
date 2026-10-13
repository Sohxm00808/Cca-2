"""
Name : Usaid Shaikh
PRN  : 130225012
Creates input_data.xlsx (sheets: Training, Input) used by traditional_vs_ml.py
Run  : python create_input_excel.py
"""
import random
import os
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

# Constants optimized for legibility and reuse
DATA_HEADERS = ["ApplicantID", "MonthlyIncome", "CreditScore", "ExistingEMI", "LoanAmount"]
HEADER_FONT = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
HEADER_FILL = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")

def generate_applicant_metrics(applicant_id: int) -> list:
    """Simulates realistic applicant financials based on randomized boundaries."""
    monthly_income = random.randrange(20000, 150000, 5000)
    credit_score = random.randrange(550, 850, 10)
    
    # Pythonic mathematical truncation using floor division
    existing_emi = int((monthly_income * random.uniform(0.0, 0.7)) // 500 * 500)
    requested_loan = int((monthly_income * random.uniform(5, 40)) // 10000 * 10000)
    
    return [applicant_id, monthly_income, credit_score, existing_emi, requested_loan]

def evaluate_historical_approval(record: list) -> str:
    """Applies legacy scoring heuristic to determine historical approval outcomes."""
    _, income, credit, emi, loan = record
    
    normalized_credit = (credit - 600) / 250
    debt_to_income_weight = (emi / income) * 0.9
    loan_to_income_weight = loan / (income * 60)
    
    composite_score = normalized_credit - debt_to_income_weight - loan_to_income_weight
    return "Yes" if composite_score > 0.05 else "No"

def format_sheet_layout(worksheet):
    """Applies professional styling, freezes headers, and auto-fits columns."""
    # Style the header row efficiently using a generator expression
    for cell in worksheet[1]:
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
    
    # Dynamic column width auto-fitting based on maximum content length
    for col in worksheet.columns:
        max_len = max(len(str(cell.value or '')) for cell in col)
        col_letter = get_column_letter(col[0].column)
        worksheet.column_dimensions[col_letter].width = max(max_len + 3, 16)

def build_dataset(output_filename="input_data.xlsx"):
    """Generates the structured Training and Input sheets with deterministic seeding."""
    random.seed(127)  # Kept strict to preserve identical data generation alignment
    wb = Workbook()
    
    # 1. Construct Training Sheet
    ws_train = wb.active
    ws_train.title = "Training"
    ws_train.append(DATA_HEADERS + ["Approved"])
    
    # Populating training logs (ID 1 to 60)
    for i in range(1, 61):
        row_data = generate_applicant_metrics(i)
        approval_status = evaluate_historical_approval(row_data)
        ws_train.append(row_data + [approval_status])
        
    # 2. Construct Input Sheet
    ws_input = wb.create_sheet(title="Input")
    ws_input.append(DATA_HEADERS)
    
    # Populating new blind application matrix (ID 101 to 115)
    for i in range(101, 116):
        ws_input.append(generate_applicant_metrics(i))
        
    # Apply standard themes and grid layouts
    format_sheet_layout(ws_train)
    format_sheet_layout(ws_input)
    
    # Save spreadsheet payload
    wb.save(output_filename)
    print(f"Successfully compiled collaborative dataset: '{output_filename}'")

if __name__ == "__main__":
    build_dataset()
