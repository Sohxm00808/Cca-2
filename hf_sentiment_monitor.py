"""
Name : abhishak raskar
PRN  : 1302250009
Topic: ML Framework - Hugging Face Sentiment Monitor (multiple problem statements)

Uses a Hugging Face pipeline when `transformers` is installed (pip install transformers torch).
If it is not available, a small keyword fallback runs so the script still works offline.
"""
import csv

try:
    from transformers import pipeline
    CLASSIFIER = pipeline("sentiment-analysis",
                          model="distilbert-base-uncased-finetuned-sst-2-english")
    MODE = "HuggingFace distilbert-sst2"
except Exception:
    CLASSIFIER = None
    MODE = "Keyword fallback (transformers not installed)"

# Expanded keyword sets for collaborator variant branch
POS = {"great", "love", "excellent", "profit", "growth", "smooth", "fast", "helpful", "surge", "gain", "secure", "seamless", "happy", "reliable", "quick"}
NEG = {"bad", "hate", "poor", "loss", "fraud", "slow", "crash", "scam", "decline", "delay", "hacked", "fail", "broken", "buggy", "error"}


def classify(text):
    if CLASSIFIER:
        r = CLASSIFIER(text[:512])[0]
        return r["label"], round(r["score"], 3)
    words = set(text.lower().replace(".", "").replace(",", "").split())
    p, n = len(words & POS), len(words & NEG)
    return ("POSITIVE", 0.6) if p >= n else ("NEGATIVE", 0.6)


# ---- Collaborator Problem Statements & Scenarios ----
PROBLEMS = {
    "E1 E-commerce & retail orders": {
        "alert_if_negative_pct": 45,
        "texts": ["Delivery was extremely fast and packaging was secure.",
                  "Item arrived broken and customer support refuses to reply.",
                  "Seamless checkout experience, loved the discounts.",
                  "Package delayed by a week with zero tracking updates."]},
    "E2 Stock trading & brokerage platform": {
        "alert_if_negative_pct": 50,
        "texts": ["Order execution is lightning fast during market open.",
                  "App crashed right when I tried to exit my position, huge loss.",
                  "Charts are smooth and indicators are very helpful.",
                  "Login error locked me out of my portfolio all morning."]},
    "E3 Digital wallet & crypto exchange": {
        "alert_if_negative_pct": 40,
        "texts": ["Easy peer-to-peer transfers with zero fees.",
                  "Withdrawal stuck in pending status for hours, buggy system.",
                  "Two-factor authentication setup is very secure and quick.",
                  "Account flagged for fraud without any explanation."]},
    "E4 Insurance claim processing": {
        "alert_if_negative_pct": 35,
        "texts": ["Claim approval was processed smoothly within 24 hours.",
                  "Endless paperwork delays and poor adjuster communication.",
                  "Very helpful helpline agent resolved my policy query instantly.",
                  "Reimbursement claim rejected due to unclear hidden clauses."]},
    "E5 Online learning management system": {
        "alert_if_negative_pct": 30,
        "texts": ["Video streaming is clear and course materials are excellent.",
                  "Server errors during quiz submissions made me fail.",
                  "Great instructor feedback on programming assignments.",
                  "Platform navigation is confusing and slow."]},
}


def main():
    print(f"Collaborator Mode: {MODE}\n")
    rows = []
    for name, cfg in PROBLEMS.items():
        results = [classify(t) for t in cfg["texts"]]
        neg = sum(1 for lab, _ in results if lab == "NEGATIVE")
        neg_pct = 100 * neg / len(results)
        alert = "ALERT" if neg_pct >= cfg["alert_if_negative_pct"] else "OK"
        print(f">> {name} | negative {neg_pct:.0f}% (threshold {cfg['alert_if_negative_pct']}%) -> {alert}")
        for t, (lab, sc) in zip(cfg["texts"], results):
            print(f"    [{lab:8} {sc}] {t}")
            rows.append([name, t, lab, sc, alert])
        print()
        
    output_file = "sentiment_report_v2.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([["Problem", "Text", "Label", "Score", "Status"]] + rows)
    print(f"Saved {output_file} successfully.")


if __name__ == "__main__":
    main()
