import re
from typing import Dict


PATTERNS = {
    "Email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "Phone": r"(?<!\d)(?:\+91[\s-]?(?:[6-9]\d{4}[\s-]?\d{5}|[6-9]\d{9})|[6-9]\d{4}[\s-]?\d{5}|\+?\d{1,3}[\s.-]?\d{3,4}[\s.-]?\d{3,4})(?!\d)",
    "LinkedIn": r"(?:https?://)?(?:www\.)?linkedin\.com/in/[A-Za-z0-9._-]+",
    "GitHub": r"(?:https?://)?(?:www\.)?github\.com/[A-Za-z0-9._-]+",
}


def extract_contact_information(text: str) -> Dict[str, str]:
    results: Dict[str, str] = {}
    for label, pattern in PATTERNS.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        results[label] = match.group(0).strip() if match else "Not detected"
    return results
