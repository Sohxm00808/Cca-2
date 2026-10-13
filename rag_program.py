"""
Name : Usaid Shaikh
PRN  : 1302250126
Topic: ML Framework - Retrieval Augmented Generation (RAG) with extra rules

Pipeline: question -> rules check -> retrieve top documents (TF-IDF) -> build grounded answer + sources.
Runs fully offline. The 'generation' step is extractive (answer built only from retrieved text),
which keeps answers grounded in the knowledge base.
"""
import re
from typing import Dict, List, Tuple, Set
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Unified Repository Configuration Management
KNOWLEDGE_BASE: Dict[str, str] = {
    "KB1-UPI": "UPI is an instant real-time payment system in India built by NPCI. Users can transfer money between bank accounts using a virtual payment address. Transactions work 24x7.",
    "KB2-CreditScore": "A credit score in India usually ranges from 300 to 900. A score above 750 is considered good. It depends on repayment history, credit utilisation and length of credit history.",
    "KB3-SIP": "A Systematic Investment Plan lets an investor put a fixed amount into a mutual fund at regular intervals. It averages the purchase cost and builds discipline.",
    "KB4-EMI": "EMI stands for equated monthly instalment. It is the fixed payment a borrower makes every month to repay a loan, covering principal and interest.",
    "KB5-KYC": "KYC means Know Your Customer. Banks verify identity and address using documents such as Aadhaar and PAN to prevent fraud and money laundering.",
    "KB6-Blockchain": "A blockchain is a distributed ledger where records are grouped into blocks linked by cryptographic hashes, making past records very hard to change.",
}


class KnowledgeRetriever:
    """Production-grade retrieval pipeline matching text constraints against configured vector indices."""
    
    MIN_QUESTION_WORDS: int = 3
    TOP_K_DOCUMENTS: int = 2
    MIN_SIMILARITY_THRESHOLD: float = 0.12
    MAX_OUTPUT_SENTENCES: int = 3
    
    FINANCIAL_ADVICE_TRIGGERS: Set[str] = {"buy", "sell", "invest in", "which stock", "guarantee"}
    PII_MASK_PATTERN: re.Pattern = re.compile(r"\b\d{10,16}\b|[\w.]+@[\w.]+")

    def __init__(self, database: Dict[str, str]):
        self.kb = database
        self.doc_ids = list(database.keys())
        
        # Build document feature extractors and fit structural matrix
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(list(database.values()))

    def mask_and_clean_query(self, raw_query: str) -> str:
        """Sanitizes text inputs by cleaning layout spaces and masking potential PII anomalies."""
        return self.PII_MASK_PATTERN.sub("[MASKED]", raw_query.strip())

    def resolve_query(self, raw_query: str) -> Tuple[str, str, List[Tuple[str, float]]]:
        """Processes incoming inquiries against security profiles and document similarity spaces."""
        cleaned_query = self.mask_and_clean_query(raw_query)
        query_tokens = cleaned_query.split()

        # Guardrail R1: Validate structural token lengths
        if len(query_tokens) < self.MIN_QUESTION_WORDS:
            return cleaned_query, "Rule R1: question too short, please ask a complete question.", []

        # Guardrail R5: Intercept speculative financial query vectors
        normalized_query = cleaned_query.lower()
        if any(trigger in normalized_query for trigger in self.FINANCIAL_ADVICE_TRIGGERS):
            return cleaned_query, "Rule R5: I cannot give personal investment advice. I can explain concepts only.", []

        # Vectorization & Scoring Phase
        query_vector = self.vectorizer.transform([cleaned_query])
        similarities = cosine_similarity(query_vector, self.matrix).flatten()
        
        # Rank source items matching criteria thresholds
        scored_docs = sorted(zip(self.doc_ids, similarities), key=lambda item: item[1], reverse=True)
        valid_sources = [(d_id, score) for d_id, score in scored_docs[:self.TOP_K_DOCUMENTS] if score >= self.MIN_SIMILARITY_THRESHOLD]

        # Guardrail R3: Yield if zero records scale above semantic cut-offs
        if not valid_sources:
            return cleaned_query, "Rule R3: no reliable source found in the knowledge base. I don't know.", []

        # Sentence-level optimization: extract target sentences safely
        sentences: List[str] = []
        for doc_id, _ in valid_sources:
            sentences.extend(re.split(r"(?<=[.!?])\s+", self.kb[doc_id]))

        # Collaborative upgrade: Vectorize candidate sentences as a single batch operation for performance
        if sentences:
            sentence_matrix = self.vectorizer.transform(sentences)
            sentence_scores = cosine_similarity(query_vector, sentence_matrix).flatten()
            ranked_sentence_indices = np.argsort(sentence_scores)[::-1][:self.MAX_OUTPUT_SENTENCES]
            response_payload = " ".join([sentences[idx] for idx in ranked_sentence_indices])
        else:
            response_payload = ""

        formatted_citations = [(doc_id, round(float(score), 2)) for doc_id, score in valid_sources]
        return cleaned_query, response_payload, formatted_citations


def execute_test_suite() -> None:
    """Runs automated functional checks across structural validation criteria."""
    engine = KnowledgeRetriever(database=KNOWLEDGE_BASE)
    
    test_cases: List[str] = [
        "What is UPI and how does it work?",
        "How is a credit score calculated?",
        "EMI?",
        "Which stock should I buy tomorrow?",
        "My phone is 9876543210, explain KYC documents please",
        "Who won the cricket world cup?",
    ]

    for test in test_cases:
        query, answer, citations = engine.resolve_query(test)
        citation_output = citations if citations else "none"
        print(f"Q: {query}\nA: {answer}\nSources: {citation_output}\n")


if __name__ == "__main__":
    execute_test_suite()
