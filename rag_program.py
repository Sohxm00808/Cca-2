"""
Name : Jaylaksh
PRN  : 1302250524
Topic: ML Framework - Retrieval Augmented Generation (RAG) with extra rules

Pipeline: question -> rules check -> retrieve top documents (TF-IDF) -> build grounded answer + sources.
Runs fully offline. The 'generation' step is extractive (answer built only from retrieved text),
which keeps answers grounded in the knowledge base.
"""
import logging
import re
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging for professional monitoring
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# --- Knowledge Base ---
KNOWLEDGE_BASE = {
    "KB1-UPI": "UPI is an instant real-time payment system in India built by NPCI. Users can transfer money between bank accounts using a virtual payment address. Transactions work 24x7.",
    "KB2-CreditScore": "A credit score in India usually ranges from 300 to 900. A score above 750 is considered good. It depends on repayment history, credit utilisation and length of credit history.",
    "KB3-SIP": "A Systematic Investment Plan lets an investor put a fixed amount into a mutual fund at regular intervals. It averages the purchase cost and builds discipline.",
    "KB4-EMI": "EMI stands for equated monthly instalment. It is the fixed payment a borrower makes every month to repay a loan, covering principal and interest.",
    "KB5-KYC": "KYC means Know Your Customer. Banks verify identity and address using documents such as Aadhaar and PAN to prevent fraud and money laundering.",
    "KB6-Blockchain": "A blockchain is a distributed ledger where records are grouped into blocks linked by cryptographic hashes, making past records very hard to change.",
}

# --- System Guardrail Rules ---
MIN_QUESTION_WORDS = 3  # R1: Reject very short questions
TOP_K = 2  # R2: Use at most 2 documents
MIN_SIMILARITY = 0.12  # R3: Below this threshold -> fallback response
MAX_ANSWER_SENTENCES = 3  # R4: Keep answers concise
ADVICE_WORDS = {
    "buy",
    "sell",
    "invest in",
    "which stock",
    "guarantee",
}  # R5: No personal investment advice
PII_PATTERN = re.compile(
    r"\b\d{10,16}\b|[\w.]+@[\w.]+"
)  # R6: Mask sensitive phone/card/account/email strings

# Initialize TF-IDF Vectorizer globally for performance
VECTORIZER = TfidfVectorizer(stop_words="english")
DOC_IDS = list(KNOWLEDGE_BASE.keys())
DOC_MATRIX = VECTORIZER.fit_transform(KNOWLEDGE_BASE.values())


def answer(question: str) -> Tuple[str, str, List[Tuple[str, float]]]:
    """Process a user question against the knowledge base applying strict rules R1-R7."""
    # R6: Mask PII
    masked_q = PII_PATTERN.sub("[MASKED]", question.strip())

    # R1: Check question length
    if len(masked_q.split()) < MIN_QUESTION_WORDS:
        return (
            masked_q,
            "Rule R1: Question too short, please ask a complete question.",
            [],
        )

    # R5: Filter investment advice queries
    if any(w in masked_q.lower() for w in ADVICE_WORDS):
        return (
            masked_q,
            "Rule R5: I cannot give personal investment advice. I can explain concepts only.",
            [],
        )

    # Compute cosine similarities
    q_vec = VECTORIZER.transform([masked_q])
    sims = cosine_similarity(q_vec, DOC_MATRIX)[0]

    # R2 & R3: Rank documents and filter by minimum similarity threshold
    ranked = sorted(
        zip(DOC_IDS, sims), key=lambda x: x[1], reverse=True
    )[:TOP_K]
    ranked = [(d, s) for d, s in ranked if s >= MIN_SIMILARITY]

    if not ranked:
        return (
            masked_q,
            "Rule R3: No reliable source found in the knowledge base. I don't know.",
            [],
        )

    # Extract sentences and select top matches (R4)
    sentences = []
    for d, _ in ranked:
        sentences += re.split(r"(?<=[.!?])\s+", KNOWLEDGE_BASE[d])

    best_sentences = sorted(
        sentences,
        key=lambda s: cosine_similarity(
            VECTORIZER.transform([masked_q]), VECTORIZER.transform([s])
        )[0][0],
        reverse=True,
    )[:MAX_ANSWER_SENTENCES]

    # R7: Return response with source citations
    formatted_sources = [(d, round(float(s), 2)) for d, s in ranked]
    return masked_q, " ".join(best_sentences), formatted_sources


if __name__ == "__main__":
    tests = [
        "What is UPI and how does it work?",
        "How is a credit score calculated?",
        "EMI?",
        "Which stock should I buy tomorrow?",
        "My phone is 9876543210, explain KYC documents please",
        "Who won the cricket world cup?",
    ]
    logging.info("Executing sample queries...\n")
    for t in tests:
        q, a, src = answer(t)
        print(f"Q: {q}")
        print(f"A: {a}")
        print(f"Sources: {src if src else 'none'}\n")
