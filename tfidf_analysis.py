from typing import Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def calculate_tfidf(resume_tokens: list[str], job_tokens: list[str]) -> Tuple[pd.DataFrame, object]:
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform([" ".join(resume_tokens), " ".join(job_tokens)])
    terms = vectorizer.get_feature_names_out()
    frame = pd.DataFrame(
        {
            "Term": terms,
            "Resume TF-IDF": matrix[0].toarray().ravel(),
            "Job Description TF-IDF": matrix[1].toarray().ravel(),
        }
    )
    frame["Combined Importance"] = frame["Resume TF-IDF"] + frame["Job Description TF-IDF"]
    return frame.sort_values("Combined Importance", ascending=False).reset_index(drop=True), matrix
