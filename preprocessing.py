import re
from typing import Dict, List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def ensure_nltk_resources() -> None:
    resources = {
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
    }
    for resource_path, package_name in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                nltk.download(package_name, quiet=True)
            except Exception:
                # The tokenizer has a regex fallback, so a network failure is non-fatal.
                continue


def preprocess_stages(text: str) -> Dict[str, object]:
    """Return each observable stage of the NLP preprocessing pipeline."""
    ensure_nltk_resources()
    lowercase = text.lower()
    cleaned = re.sub(r"[^a-z0-9+#.\s-]", " ", lowercase)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    try:
        raw_tokens = word_tokenize(cleaned)
    except LookupError:
        raw_tokens = re.findall(r"[a-z0-9]+(?:[+#.]?[a-z0-9]+)*", cleaned)

    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        stop_words = set()
    filtered_tokens: List[str] = []
    for token in raw_tokens:
        token = token.strip(".-")
        if not token or token in stop_words or not re.search(r"[a-z0-9]", token):
            continue
        filtered_tokens.append(token)

    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens: List[str] = []
    for token in filtered_tokens:
        try:
            token = lemmatizer.lemmatize(token)
        except LookupError:
            pass
        lemmatized_tokens.append(token)

    return {
        "original": text,
        "lowercase": lowercase,
        "cleaned": cleaned,
        "tokens": raw_tokens,
        "without_stopwords": filtered_tokens,
        "lemmatized": lemmatized_tokens,
    }


def preprocess_text(text: str) -> List[str]:
    """Lowercase, clean, tokenize, remove stop words, and lemmatize text."""
    return list(preprocess_stages(text)["lemmatized"])
