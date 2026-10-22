"""
Name : Jaylaksh
PRN  : 1302250524
Topic: ML Framework - Retrieval Augmented Generation (RAG) with extra rules

Pipeline: question -> rules check -> retrieve top documents (TF-IDF) -> build grounded answer + sources.
Runs fully offline. The 'generation' step is extractive (answer built only from retrieved text),
which keeps answers grounded in the knowledge base.
"""
import argparse
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KNOWLEDGE_BASE = {
    "KB1-UPI": "UPI is an instant real-time payment system in India built by NPCI. Users can transfer money between bank accounts using a virtual payment address. Transactions work 24x7.",
    "KB2-CreditScore": "A credit score in India usually ranges from 300 to 900. A score above 750 is considered good. It depends on repayment history, credit utilisation and length of credit history.",
    "KB3-SIP": "A Systematic Investment Plan lets an investor put a fixed amount into a mutual fund at regular intervals. It averages the purchase cost and builds discipline.",
    "KB4-EMI": "EMI stands for equated monthly instalment. It is the fixed payment a borrower makes every month to repay a loan, covering principal and interest.",
    "KB5-KYC": "KYC means Know Your Customer. Banks verify identity and address using documents such as Aadhaar and PAN to prevent fraud and money laundering.",
    "KB6-Blockchain": "A blockchain is a distributed ledger where records are grouped into blocks linked by cryptographic hashes, making past records very hard to change.",
    "KB7-MutualFunds": "Mutual funds pool money from multiple investors to invest in a diversified portfolio of stocks, bonds, or other securities managed by professional fund managers."
}

# ------------------------- RULES -------------------------
MIN_QUESTION_WORDS = 3
TOP_K = 2
MIN_SIMILARITY = 0.12
MAX_ANSWER_SENTENCES = 3
ADVICE_WORDS = {"buy", "sell", "invest in", "which stock", "guarantee", "recommend"}
PII_PATTERN = re.compile(r"\b\d{10,16}\b|[\w.]+@[\w.]+")

vec = TfidfVectorizer(stop_words="english")
DOC_IDS = list(KNOWLEDGE_BASE)
MATRIX = vec.fit_transform(KNOWLEDGE_BASE.values())


def answer_query(question):
    q = PII_PATTERN.sub("[MASKED]", question.strip())  # R6
    if len(q.split()) < MIN_QUESTION_WORDS:            # R1
        return q, "Rule R1: Question too short, please ask a complete question.", []
    if any(w in q.lower() for w in ADVICE_WORDS):        # R5
        return q, "Rule R5: I cannot give personal investment advice. I can explain concepts only.", []

    sims = cosine_similarity(vec.transform([q]), MATRIX)[0]
    ranked = sorted(zip(DOC_IDS, sims), key=lambda x: x[1], reverse=True)[:TOP_K]   # R2
    ranked = [(d, s) for d, s in ranked if s >= MIN_SIMILARITY]             # R3
    if not ranked:
        return q, "Rule R3: No reliable source found in the knowledge base. I don't know.", []

    sentences = []
    for d, _ in ranked:
        sentences += re.split(r"(?<=[.!?])\s+", KNOWLEDGE_BASE[d])
    best = sorted(sentences, key=lambda s: cosine_similarity(vec.transform([q]), vec.transform([s]))[0][0],
                  reverse=True)[:MAX_ANSWER_SENTENCES]                     # R4
    return q, " ".join(best), [(d, round(float(s), 2)) for d, s in ranked]  # R7


def run_tests(export_json=True):
    tests = [
        "What is UPI and how does it work?",
        "How is a credit score calculated?",
        "EMI?",
        "Which stock should I buy tomorrow?",
        "My phone is 9876543210, explain KYC documents please",
        "Who won the cricket world cup?",
        "Tell me about mutual funds portfolio diversification"
    ]
    
    report = []
    print("Running Collaborator RAG Test Suite:\n" + "="*50)
    for i, t in enumerate(tests, 1):
        q, a, src = answer_query(t)
        print(f"[{i}] Q: {q}")
        print(f"    A: {a}")
        print(f"    Sources: {src if src else 'none'}\n" + "-"*50)
        report.append({"id": i, "question": q, "answer": a, "sources": src})
        
    if export_json:
        with open("rag_evaluation_results.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
        print("Successfully exported test evaluation to 'rag_evaluation_results.json'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run local TF-IDF RAG Pipeline with safety guardrails.")
    parser.add_argument("--query", type=str, help="Ask a single question directly via command line")
    args = parser.parse_args()

    if args.query:
        q, a, src = answer_query(args.query)
        print(f"\nQ: {q}\nA: {a}\nSources: {src if src else 'none'}\n")
    else:
        run_tests()
