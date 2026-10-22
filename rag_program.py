"""
Name : abhishak raskar
PRN  : 1302250009
Topic: ML Framework - Retrieval Augmented Generation (RAG) with extra rules

Pipeline: question -> rules check -> retrieve top documents (TF-IDF) -> build grounded answer + sources.
Runs fully offline. The 'generation' step is extractive (answer built only from retrieved text),
which keeps answers grounded in the knowledge base.
"""
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KNOWLEDGE_BASE = {
    "KB1-Cloud": "Cloud computing is the delivery of computing services over the internet, including storage, servers, databases, and networking.",
    "KB2-API": "An API or Application Programming Interface is a set of rules that allows different software applications to communicate with each other.",
    "KB3-Docker": "Docker is a tool designed to make it easier to create, deploy, and run applications by packaging them into standardized containers.",
    "KB4-Microservices": "Microservices architecture is an approach to software development where an application is built as a collection of loosely coupled services.",
    "KB5-CICD": "CI/CD stands for Continuous Integration and Continuous Deployment, automating the building, testing, and releasing of software updates.",
    "KB6-Security": "Cybersecurity involves protecting systems, networks, and programs from digital attacks, unauthorized access, and data breaches.",
}

# ------------------------- RULES -------------------------
MIN_QUESTION_WORDS = 2          # R1: reject very short questions
TOP_K = 3                       # R2: use at most 3 documents
MIN_SIMILARITY = 0.10           # R3: below this -> "I don't know"
MAX_ANSWER_SENTENCES = 2        # R4: keep answers concise (max 2 sentences)
ADVICE_WORDS = {"hack into", "bypass security", "ddos", "crack password", "exploit"}  # R5: restrict harmful requests
PII_PATTERN = re.compile(r"\b\d{10,16}\b|[\w.]+@[\w.]+")                  # R6: mask phone/card/account/email

vec = TfidfVectorizer(stop_words="english")
DOC_IDS = list(KNOWLEDGE_BASE)
MATRIX = vec.fit_transform(KNOWLEDGE_BASE.values())


def answer(question):
    q = PII_PATTERN.sub("[MASKED]", question.strip())                        # R6
    if len(q.split()) < MIN_QUESTION_WORDS:                                  # R1
        return q, "Rule R1: question too short, please ask a complete question.", []
    if any(w in q.lower() for w in ADVICE_WORDS):                            # R5
        return q, "Rule R5: I cannot assist with malicious actions or security exploits.", []

    sims = cosine_similarity(vec.transform([q]), MATRIX)[0]
    ranked = sorted(zip(DOC_IDS, sims), key=lambda x: x[1], reverse=True)[:TOP_K]    # R2
    ranked = [(d, s) for d, s in ranked if s >= MIN_SIMILARITY]              # R3
    if not ranked:
        return q, "Rule R3: no reliable source found in the knowledge base. I don't know.", []

    sentences = []
    for d, _ in ranked:
        sentences += re.split(r"(?<=[.!?])\s+", KNOWLEDGE_BASE[d])
    best = sorted(sentences, key=lambda s: cosine_similarity(vec.transform([q]), vec.transform([s]))[0][0],
                  reverse=True)[:MAX_ANSWER_SENTENCES]                       # R4
    return q, " ".join(best), [(d, round(float(s), 2)) for d, s in ranked]  # R7: cite sources


if __name__ == "__main__":
    tests = [
        "What is cloud computing and why use it?",
        "Explain API integration",
        "Docker?",
        "How to hack into a remote server?",
        "My email is dev@test.com, tell me about CI/CD pipelines",
        "Who invented the telephone?",
    ]
    for t in tests:
        q, a, src = answer(t)
        print(f"Q: {q}\nA: {a}\nSources: {src if src else 'none'}\n")
