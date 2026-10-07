from collections import Counter
from typing import Iterable


def calculate_word_frequency(tokens: Iterable[str], top_n: int = 15) -> Counter:
    return Counter(tokens).most_common(top_n)
