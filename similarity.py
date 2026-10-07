from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarity(resume_vector, job_vector) -> float:
    return float(cosine_similarity(resume_vector, job_vector)[0][0])
