"""
Name : Usaid Shaikh
PRN  : 1302250126
Topic: Traditional Programming vs Machine Learning (Loan Approval)

Traditional programming : humans write the RULES, the program applies them to data.
Machine learning        : we give DATA + ANSWERS (Training sheet), the model learns the rules.

Input  : input_data.xlsx  -> sheets 'Training' (with Approved) and 'Input' (new applicants)
Output : written back to the same file, sheet 'Output'
"""
import logging
from typing import Tuple, List
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class UnderwritingEngine:
    """Production-grade framework evaluating applications via both heuristics and ML models."""

    def __init__(self, data_path: str = "input_data.xlsx"):
        self.data_path = data_path
        self.features: List[str] = ["MonthlyIncome", "CreditScore", "ExistingEMI", "LoanAmount"]
        self.model = DecisionTreeClassifier(max_depth=4, random_state=1)
        self.test_accuracy: float = 0.0

    def evaluate_heuristics(self, df: pd.DataFrame) -> pd.Series:
        """Vectorized execution of legacy policy rules for maximum data performance."""
        # Baseline rejection criteria array masks
        bad_credit = df["CreditScore"] < 700
        overextended_debt = df["ExistingEMI"] > (0.40 * df["MonthlyIncome"])
        excessive_leverage = df["LoanAmount"] > (20 * df["MonthlyIncome"])

        # Construct consolidated verification array
        rejection_mask = bad_credit | overextended_debt | excessive_leverage
        return pd.Series(
            ["No" if rejected else "Yes" for rejected in rejection_mask],
            index=df.index
        )

    def train_ml_pipeline(self, train_df: pd.DataFrame) -> None:
        """Splits evaluation logs, checks metrics, and fits model to full operational scope."""
        X = train_df[self.features]
        y = train_df["Approved"]

        # Metric evaluation split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
        
        # Benchmarking validation metrics
        validation_model = DecisionTreeClassifier(max_depth=4, random_state=1)
        validation_model.fit(X_train, y_train)
        predictions = validation_model.predict(X_test)
        self.test_accuracy = accuracy_score(y_test, predictions)

        # Refit deployment model on complete data matrix
        self.model.fit(X, y)
        logging.info(f"Machine Learning pipeline trained successfully. Held-out validation Accuracy: {self.test_accuracy:.2%}")

    def execute_pipeline(self) -> None:
        """Loads sheet parameters, correlates scoring channels, and appends the final matrix."""
        try:
            train_df = pd.read_excel(self.data_path, sheet_name="Training")
            input_df = pd.read_excel(self.data_path, sheet_name="Input")
        except FileNotFoundError:
            logging.error(f"Target data spreadsheet payload not found at path: {self.data_path}")
            return
        except ValueError as err:
            logging.error(f"Spreadsheet sheet matrix parsing anomaly encountered: {err}")
            return

        # Initialize output dataframe configuration
        output_df = input_df.copy()
        
        # Parallel evaluations
        output_df["Traditional_Decision"] = self.evaluate_heuristics(input_df)
        self.train_ml_pipeline(train_df)
        output_df["ML_Decision"] = self.model.predict(input_df[self.features])
        
        # Structural mapping check
        decisions_match = output_df["Traditional_Decision"] == output_df["ML_Decision"]
        output_df["Same_Result"] = decisions_mask = decisions_match.map({True: "Yes", False: "No"})

        # Export to existing file workspace safely
        try:
            with pd.ExcelWriter(self.data_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
                output_df.to_excel(writer, sheet_name="Output", index=False)
            logging.info("Successfully pushed combined analysis payload back to database engine.")
        except IOError as err:
            logging.error(f"Writing outputs back to dataset blocked by lock anomalies: {err}")
            return

        # Terminal reporting printout
        match_count = (decisions_mask == "Yes").sum()
        print(f"\nML test accuracy on held-out training data: {self.test_accuracy:.2%}")
        print(f"Traditional vs ML agree on {match_count}/{len(output_df)} applicants\n")
        print(output_df.to_string(index=False))


if __name__ == "__main__":
    engine = UnderwritingEngine()
    engine.execute_pipeline()
