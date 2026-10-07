SKILL_VOCABULARY = {
    "Programming": ["Python", "Java", "C", "C++", "JavaScript"],
    "Data": ["SQL", "MySQL", "PostgreSQL", "MongoDB", "Pandas", "NumPy", "Data Analysis", "Data Science"],
    "AI / ML": [
        "Machine Learning",
        "Deep Learning",
        "Artificial Intelligence",
        "Natural Language Processing",
        "NLP",
        "Computer Vision",
        "TensorFlow",
        "PyTorch",
        "Scikit-learn",
    ],
    "Web": ["HTML", "CSS", "JavaScript", "React", "Node.js", "Flask", "Django"],
    "Tools": ["Git", "GitHub", "Docker", "Linux", "Problem Solving"],
}

ALL_SKILLS = [skill for skills in SKILL_VOCABULARY.values() for skill in skills]

SKILL_ALIASES = {
    "ML": "Machine Learning",
    "NLP": "Natural Language Processing",
    "CV": "Computer Vision",
    "SCIKIT LEARN": "Scikit-learn",
    "NODE JS": "Node.js",
}

SKILL_TERMS = ALL_SKILLS + list(SKILL_ALIASES)

GENERIC_TFIDF_EXCLUSIONS = {
    "candidate", "job", "looking", "required", "preferred", "good", "strong",
    "comfortable", "advantage", "experience", "knowledge", "working", "work",
    "team", "communication", "role", "position", "application", "responsibility",
    "based", "engineer", "title", "building", "developing", "problem", "solving", "ml", "nlp", "cv",
}
