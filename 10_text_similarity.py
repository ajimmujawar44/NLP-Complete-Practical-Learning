"""
12 - TEXT SIMILARITY
======================
WHAT  : Measuring how similar two pieces of text are, numerically.
        - Jaccard Similarity: overlap of unique words / union of unique words.
        - Cosine Similarity (on TF-IDF vectors): angle between two text
          vectors — the industry standard for document similarity.
WHERE : Plagiarism detection, duplicate question detection (Quora),
        search engine ranking (how relevant is a doc to a query),
        recommendation systems ("similar articles").
WHY   : Comparing raw strings ("do these match exactly?") is too rigid.
        Similarity scores let us compare MEANING/OVERLAP even when exact
        wording differs.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """intersection(words) / union(words) — simple and interpretable."""
    set_a = set(text_a.lower().split())
    set_b = set(text_b.lower().split())
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


def cosine_similarity_tfidf(text_a: str, text_b: str) -> float:
    """Vectorizes both texts with TF-IDF, then computes cosine similarity."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text_a, text_b])
    score = sk_cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return score[0][0]


if __name__ == "__main__":
    pairs = [
        ("I love machine learning", "I love deep learning"),
        ("The cat sat on the mat", "A dog is running in the park"),
        ("How to learn Python programming", "How can I learn Python coding"),
        ("NLP is a subset of AI", "NLP is a subset of artificial intelligence"),
    ]

    print(f"{'Text A':<40}{'Text B':<40}{'Jaccard':<10}{'Cosine (TF-IDF)'}")
    print("-" * 105)
    for a, b in pairs:
        jac = jaccard_similarity(a, b)
        cos = cosine_similarity_tfidf(a, b)
        print(f"{a[:37]:<40}{b[:37]:<40}{jac:<10.2f}{cos:.2f}")

    print("\nNOTE: For SEMANTIC similarity (understanding synonyms like")
    print("'coding' ~ 'programming'), use word/sentence embeddings instead")
    print("of TF-IDF — see modules 07/08, or Sentence-BERT in module 16.")
