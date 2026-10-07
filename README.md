# Job Description & Resume Matching System

Group 9 academic NLP project built with Python and Streamlit. It compares the actual resume and job description entered by the user using transparent statistical NLP: preprocessing, regex extraction, word frequency, TF-IDF, cosine similarity, and vocabulary-based skill comparison.

## Features

- Paste or upload `.txt` resume and job-description text.
- Lowercase, punctuation cleaning, whitespace normalization, NLTK tokenization, stop-word removal, and WordNet lemmatization.
- Regex extraction for email, phone, LinkedIn, and GitHub.
- Dynamic word-frequency charts and TF-IDF tables.
- Independently reported TF-IDF text similarity and skill match percentages.
- Common, missing, and additional skills from the editable vocabulary in `config/skills.py`.
- Required and preferred skill cues when a priority phrase occurs in the same job-description sentence.
- Separate common vocabulary and filtered important TF-IDF terms so generic recruitment language is not mislabeled as a skill.
- Stage-by-stage preprocessing demonstration and selectable Top 5, 10, 15, or 20 frequency views.
- Apple-inspired glass materials, restrained color, clear hierarchy, direct feedback, and reduced-motion CSS.

## Installation and running

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the NLTK resources required for tokenization, stop words, and WordNet when they are unavailable locally.

## Project structure

```text
app.py
preprocessing.py
regex_extractor.py
frequency_analysis.py
tfidf_analysis.py
similarity.py
skill_matcher.py
config/skills.py
requirements.txt
```

## Methodology

The app validates both inputs, shows lowercase, cleaned, tokenized, stop-word-filtered, and lemmatized stages, extracts contact information and known skills, calculates term frequencies, builds TF-IDF vectors, calculates cosine similarity, and compares detected skills. The two scores are intentionally separate: cosine similarity is textual overlap, while skill match is the fraction of job-description skills also found in the resume. Priority labels are conservative and only apply when a cue such as `required` or `preferred` occurs in the same sentence as a detected skill.

## Limitations

Skill detection depends on the predefined vocabulary. Exact wording and synonyms affect matching. TF-IDF does not understand complete semantic meaning, and cosine similarity does not measure job suitability. Regex extraction depends on recognizable patterns. The system is not a replacement for human recruitment decisions and does not use a supervised training dataset or model.

## Sample usage

Paste any real resume and job description into the two input areas, select **Analyze resume**, and inspect the Overview, Resume information, Preprocessing, Frequency, TF-IDF, Skill analysis, and Methodology tabs. Changing either document and analyzing again recalculates every result.
