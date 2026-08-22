"""
11 - SPAM DETECTION
=====================
WHAT  : A binary text classification project (Spam vs Ham/Not-Spam) —
        applies everything from modules 02, 06, and 09 to a real problem.
        Pipeline: clean text -> TF-IDF vectorize -> Naive Bayes classifier.
WHERE : Email providers (Gmail), SMS filtering, comment moderation on
        YouTube/Instagram, review-bombing detection.
WHY   : Naive Bayes is the classic go-to algorithm for spam detection
        because it works great with word-frequency features and is very
        fast to train — this is the exact type of model Gmail used early on.
"""

import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Toy labeled dataset: 1 = Spam, 0 = Ham (legit message)
MESSAGES = [
    "WIN a free iPhone now!!! Click this link to claim your prize",
    "Hey, are we still meeting for lunch tomorrow?",
    "URGENT: Your account has been suspended, verify now at bit.ly/xyz",
    "Can you send me the report before 5 PM?",
    "Congratulations! You have won $1000 cash, reply YES to claim",
    "Mom, I'll be home late tonight, don't wait for dinner",
    "Free entry in a weekly contest, text WIN to 80086",
    "Let's catch up this weekend, it's been a while",
    "Limited time offer! Buy 1 get 1 free, click here now",
    "The meeting has been rescheduled to Monday 10 AM",
    "You have been selected for a cash reward, claim within 24 hrs",
    "Please review the attached document and share feedback",
]
LABELS = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"https?://\S+|bit\.ly/\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def build_spam_classifier(messages, labels):
    cleaned = [clean_text(m) for m in messages]

    X_train, X_test, y_train, y_test = train_test_split(
        cleaned, labels, test_size=0.3, random_state=42, stratify=labels
    )

    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_test_vec)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("\nConfusion Matrix (rows=actual, cols=predicted) [Ham, Spam]:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=["Ham", "Spam"], zero_division=0))

    return model, vectorizer


def predict_message(model, vectorizer, message):
    cleaned = clean_text(message)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    probability = model.predict_proba(vec)[0]
    label = "SPAM" if prediction == 1 else "HAM (not spam)"
    return label, max(probability)


if __name__ == "__main__":
    model, vectorizer = build_spam_classifier(MESSAGES, LABELS)

    print("\n--- Testing on new messages ---")
    test_messages = [
        "Claim your free prize now, limited slots available!",
        "Are you coming to the office tomorrow?",
    ]
    for msg in test_messages:
        label, confidence = predict_message(model, vectorizer, msg)
        print(f"  '{msg}' -> {label} (confidence={confidence:.2f})")
