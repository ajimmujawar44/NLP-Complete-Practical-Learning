"""
10 - SENTIMENT ANALYSIS
=========================
WHAT  : Detecting the emotional tone of text — Positive, Negative, or
        Neutral. Two common approaches:
        1. LEXICON-based (rule-based): uses a pre-built dictionary of
           words tagged with sentiment scores (e.g. VADER). No training
           needed — fast and interpretable.
        2. ML-based: train a classifier (like module 09) on labeled
           positive/negative examples.
WHERE : Product review analysis (Amazon/Flipkart), social media
        monitoring (brand reputation on Twitter/X), customer feedback,
        stock market sentiment from news.
WHY   : Businesses need to know how customers FEEL at scale — reading
        millions of reviews manually is impossible; sentiment analysis
        automates it.
"""

import nltk
nltk.download("vader_lexicon", quiet=True)
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

REVIEWS = [
    "I absolutely love this product, it exceeded my expectations!",
    "This is the worst purchase I've ever made, total waste of money.",
    "The product is okay, does what it says, nothing special.",
    "Amazing quality and super fast delivery, highly recommend!",
    "Terrible customer service, I'm never buying from here again.",
]


def vader_sentiment(text: str):
    """
    VADER (Valence Aware Dictionary and sEntiment Reasoner) is tuned for
    social-media-style text (handles emojis, CAPS, punctuation like '!!!').
    Returns a compound score from -1 (very negative) to +1 (very positive).
    """
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return label, compound


def textblob_sentiment(text: str):
    """
    TextBlob gives Polarity (-1 to 1: negative to positive) and
    Subjectivity (0 to 1: objective fact to subjective opinion).
    """
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity


if __name__ == "__main__":
    print(f"{'Review':<65}{'VADER':<12}{'Score':<8}{'TB Polarity'}")
    print("-" * 100)
    for review in REVIEWS:
        label, score = vader_sentiment(review)
        polarity, subjectivity = textblob_sentiment(review)
        short = (review[:60] + "...") if len(review) > 60 else review
        print(f"{short:<65}{label:<12}{score:<8.2f}{polarity:.2f}")

    print("\nNOTE: For domain-specific accuracy (e.g. sarcasm, industry jargon),")
    print("train a custom ML classifier using the pipeline from module 09,")
    print("or fine-tune a BERT model (module 16) on labeled sentiment data.")
