"""
Name : Usaid Shaikh
PRN  : 1302250126
Topic: ML Framework - Hugging Face Sentiment Monitor (multiple problem statements)

Uses a Hugging Face pipeline when `transformers` is installed (pip install transformers torch).
If it is not available, a small keyword fallback runs so the script still works offline.
"""
import csv
import re
from typing import Dict, List, Tuple, Any

# Attempt model instantiation using modern transformers pipeline
try:
    from transformers import pipeline
    _PIPELINE = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    EXECUTION_MODE = "Transformer (distilbert-sst2)"
except Exception:
    _PIPELINE = None
    EXECUTION_MODE = "Heuristic Fallback (Regex Tokenizer)"

# Lexicons optimized for quick scoring fallback
LEXICON_POS = {"great", "love", "excellent", "profit", "growth", "smooth", "fast", "helpful", "surge", "gain", "secure"}
LEXICON_NEG = {"bad", "hate", "poor", "loss", "fraud", "slow", "crash", "scam", "decline", "delay", "hacked", "fail"}


class SentimentAnalyzer:
    """Handles textual classification using either deep learning models or structural fallback rules."""

    @staticmethod
    def evaluate(text: str) -> Tuple[str, float]:
        """Classifies text string and returns a matching tuple of (Label, Confidence)."""
        if _PIPELINE:
            # Enforce max length constraint safely via slicing
            prediction = _PIPELINE(text[:512])[0]
            return prediction["label"].upper(), round(prediction["score"], 3)
        
        # Collaborative upgrade: Regex tokenization avoids string replacement edge cases
        tokens = set(re.findall(r'\b\w+\b', text.lower()))
        
        positive_matches = len(tokens & LEXICON_POS)
        negative_matches = len(tokens & LEXICON_NEG)
        
        if positive_matches >= negative_matches:
            return "POSITIVE", 0.600
        return "NEGATIVE", 0.600


# Structured metrics pipeline data
MONITORING_SUITES: Dict[str, Dict[str, Any]] = {
    "P1 Financial news monitoring": {
        "threshold": 50,
        "payload": [
            "Sensex surge as banking stocks post strong profit growth.",
            "Fintech startup hit by fraud scandal, shares crash.",
            "RBI keeps rates unchanged, markets stay calm and stable.",
            "Quarterly loss widens as loan defaults decline recovery."
        ]
    },
    "P2 UPI / payment app reviews": {
        "threshold": 40,
        "payload": [
            "Payments are fast and the app is smooth, love it.",
            "Transaction failed twice and refund was a long delay.",
            "Excellent cashback offers and secure login.",
            "Customer care is poor, money stuck for days."
        ]
    },
    "P3 Bank customer support tickets": {
        "threshold": 30,
        "payload": [
            "Thanks, the agent was very helpful and solved my issue.",
            "Third time complaining, nobody replies, really bad service.",
            "Card blocked after suspected fraud, fix this slow process.",
            "Loan approval was quick and the process was great."
        ]
    },
    "P4 Crypto / social media chatter": {
        "threshold": 60,
        "payload": [
            "Bitcoin gain continues, community feels bullish and excited.",
            "Exchange hacked, users panic and withdraw everything.",
            "Another scam token rug pulled investors overnight.",
            "Regulation clarity is a great step for the industry."
        ]
    },
    "P5 Campus / course feedback": {
        "threshold": 35,
        "payload": [
            "The FinTech lectures were excellent and practical.",
            "Timetable clashes and poor lab systems, very bad experience.",
            "Python sessions were helpful and well paced.",
            "Assignment deadlines are too tight."
        ]
    }
}


def run_pipeline(output_path: str = "sentiment_report.csv") -> None:
    """Executes evaluation lifecycle across all modules and exports insights to CSV."""
    print(f"Initialization Engine Mode: {EXECUTION_MODE}\n")
    report_rows: List[List[Any]] = []
    
    for tracking_scope, dataset in MONITORING_SUITES.items():
        texts = dataset["payload"]
        alert_threshold = dataset["threshold"]
        
        # Execute evaluations
        scored_records = [SentimentAnalyzer.evaluate(txt) for txt in texts]
        
        # Analytical computations
        negative_count = sum(1 for label, _ in scored_records if label == "NEGATIVE")
        negative_ratio_pct = (negative_count / len(texts)) * 100
        
        alert_status = "ALERT" if negative_ratio_pct >= alert_threshold else "OK"
        print(f"== {tracking_scope} | Negative: {negative_ratio_pct:.0f}% (Threshold: {alert_threshold}%) -> {alert_status}")
        
        for raw_text, (inferred_label, confidence_score) in zip(texts, scored_records):
            print(f"   [{inferred_label:<8} {confidence_score:.3f}] {raw_text}")
            report_rows.append([tracking_scope, raw_text, inferred_label, confidence_score, alert_status])
        print()

    # Save outputs safely with clean data streams
    try:
        with open(output_path, mode="w", newline="", encoding="utf-8") as target_file:
            csv_writer = csv.writer(target_file)
            csv_writer.writerow(["Problem", "Text", "Label", "Score", "Status"])
            csv_writer.writerows(report_rows)
        print(f"Successfully exported data to '{output_path}'")
    except IOError as err:
        print(f"Error persisting output logs to disk: {err}")


if __name__ == "__main__":
    run_pipeline()
