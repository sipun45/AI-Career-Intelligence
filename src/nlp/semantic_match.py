import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


print("Loading semantic model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Semantic model loaded successfully!")


def calculate_similarity(resume_text, job_description):

    resume_embedding = model.encode([resume_text])

    job_embedding = model.encode([job_description])

    similarity = cosine_similarity(
        resume_embedding,
        job_embedding
    )[0][0]

    match_percentage = similarity * 100

    return round(float(match_percentage), 2)


if __name__ == "__main__":

    resume = """
    Python developer with experience in machine learning,
    SQL, data analysis and deep learning.
    """

    job_description = """
    We are looking for an ML Engineer with Python,
    SQL, machine learning, deep learning and cloud skills.
    """

    score = calculate_similarity(
        resume,
        job_description
    )

    print("Resume-Job Match:", score, "%")