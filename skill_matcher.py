import re
from typing import Iterable, Set

from config.skills import ALL_SKILLS, SKILL_ALIASES, SKILL_TERMS


def _canonical_skill(skill: str) -> str:
    return SKILL_ALIASES.get(skill.upper(), skill)


def _skill_pattern(skill: str) -> str:
    parts = re.split(r"[\s-]+", skill.strip())
    expression = r"[\s/_-]+".join(re.escape(part) for part in parts if part)
    return r"(?<![a-z0-9])" + expression + r"(?![a-z0-9])"


def extract_skills(text: str, vocabulary: Iterable[str] = ALL_SKILLS) -> Set[str]:
    matches = []
    vocabulary = list(vocabulary)
    terms = SKILL_TERMS if set(vocabulary) == set(ALL_SKILLS) else vocabulary
    for skill in sorted(set(terms), key=len, reverse=True):
        matches.extend((match.start(), match.end(), _canonical_skill(skill)) for match in re.finditer(_skill_pattern(skill), text, flags=re.IGNORECASE))

    selected = []
    occupied = set()
    for start, end, skill in sorted(matches, key=lambda item: (-(item[1] - item[0]), item[0])):
        span = set(range(start, end))
        if not span & occupied:
            selected.append(skill)
            occupied.update(span)
    return set(selected)


def classify_skill_priority(text: str, skills: Set[str]) -> dict[str, Set[str]]:
    """Classify skills only when a priority cue occurs in the same sentence."""
    priority = {"required": set(), "preferred": set(), "unclassified": set()}
    sentences = re.split(r"(?<=[.!?\n])\s+", text)
    required_cues = re.compile(r"\b(required|must\s+have|mandatory|strong\s+knowledge\s+of)\b", re.IGNORECASE)
    preferred_cues = re.compile(r"\b(preferred|nice\s+to\s+have|advantage|plus)\b", re.IGNORECASE)
    for skill in skills:
        aliases = [term for term in SKILL_TERMS if _canonical_skill(term) == skill]
        skill_pattern = re.compile("|".join(_skill_pattern(term) for term in aliases), re.IGNORECASE)
        matching_sentences = [sentence for sentence in sentences if skill_pattern.search(sentence)]
        if any(required_cues.search(sentence) for sentence in matching_sentences):
            priority["required"].add(skill)
        elif any(preferred_cues.search(sentence) for sentence in matching_sentences):
            priority["preferred"].add(skill)
        else:
            priority["unclassified"].add(skill)
    return priority


def compare_skills(resume_skills: Set[str], job_skills: Set[str]) -> dict[str, Set[str]]:
    return {
        "common": resume_skills & job_skills,
        "missing": job_skills - resume_skills,
        "additional": resume_skills - job_skills,
    }
