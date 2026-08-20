"""
09 - TEXT CLASSIFICATION
==========================
WHAT  : Assigning a predefined category/label to a piece of text.
        Pipeline: raw text -> preprocess -> vectorize (TF-IDF) -> ML model
        (Naive Bayes / Logistic Regression / SVM) -> predicted label.
WHERE : Spam filters, sentiment analysis, news categorization (Sports/
        Politics/Tech), support-ticket routing, language detection.
WHY   : This is THE most common real-world NLP task. Once you understand
        this pipeline, sentiment analysis (10) and spam detection (11)
        are just this same pattern applied to a specific dataset.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Toy dataset: text + label (1 = Technology, 0 = Sports)
TEXTS = [
    "The new smartphone has an amazing camera and fast processor",
    "The team scored a goal in the last minute of the match",
    "Artificial intelligence is transforming the software industry",
    "The player broke the world record in the 100m sprint",
    "The new laptop features a powerful GPU for gaming and AI",
    "The football championship final was thrilling to watch",
    "Cloud computing helps startups scale their applications",
    "The basketball team won the tournament after overtime",
    "The latest AI chip improves machine learning performance",
    "The tennis player won the grand slam title this year",
]
LABELS = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1=Technology, 0=Sports
LABEL_NAMES = {0: "Sports", 1: "Technology"}


def train_classifier(texts, labels, model_type="naive_bayes"):
    """
    Full pipeline: TF-IDF vectorization + classifier training.
    model_type: 'naive_bayes' (fast, great baseline for text) or 'logistic_regression'
    """
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=42
    )

    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)  # IMPORTANT: transform, not fit_transform

    if model_type == "naive_bayes":
        model = MultinomialNB()
    else:
        model = LogisticRegression(max_iter=1000)

    model.fit(X_train_vec, y_train)
    predictions = model.predict(X_test_vec)

    print(f"Model: {model_type}")
    print("Accuracy:", accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions, zero_division=0))

    return model, vectorizer


def predict_new_text(model, vectorizer, text):
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    return LABEL_NAMES.get(prediction, prediction)


if __name__ == "__main__":
    model, vectorizer = train_classifier(TEXTS, LABELS, model_type="naive_bayes")

    print("\n--- Testing on brand-new sentences ---")
    new_sentences = [
        "The new gaming console has incredible graphics",
        "The athlete trained hard before the marathon",
    ]
    for sentence in new_sentences:
        label = predict_new_text(model, vectorizer, sentence)
        print(f"  '{sentence}' -> {label}")
