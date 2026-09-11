import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.nlp.text_preprocessor import clean_text

# Load Transformer once
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def create_embedding(text: str):

    cleaned_text = clean_text(text)

    embedding = model.encode(
        [cleaned_text]
    )

    return embedding


def calculate_similarity(
    resume_text: str,
    job_description: str
):

    resume_embedding = create_embedding(
        resume_text
    )

    job_embedding = create_embedding(
        job_description
    )

    similarity = cosine_similarity(
        resume_embedding,
        job_embedding
    )[0][0]

    return round(
        float(similarity * 100),
        2
    )


if __name__ == "__main__":

    resume = """
    Python developer with experience in
    SQL, machine learning, deep learning
    and cloud technologies.
    """

    job = """
    We are looking for an ML Engineer
    with experience in Python, SQL,
    machine learning, deep learning
    and cloud technologies.
    """

    score = calculate_similarity(
        resume,
        job
    )

    print(
        "Resume ↔ Job Semantic Match:",
        score,
        "%"
    )