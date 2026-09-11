import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from sentence_transformers import SentenceTransformer
import numpy as np

from text_preprocessor import clean_text


# Load Transformer model
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def create_job_embedding(job_description: str):

    # Clean job description
    cleaned_text = clean_text(
        job_description
    )

    # Create embedding
    embedding = model.encode(
        cleaned_text
    )

    return {
        "cleaned_text": cleaned_text,
        "embedding": np.array(
            embedding
        )
    }


if __name__ == "__main__":

    job_description = """
    ML Engineer

    We are looking for an ML Engineer
    with experience in Python, SQL,
    machine learning, deep learning,
    PyTorch and cloud technologies.

    The candidate should have experience
    building and deploying machine
    learning models.
    """

    result = create_job_embedding(
        job_description
    )

    print("========== CLEAN JOB DESCRIPTION ==========")

    print(
        result["cleaned_text"]
    )

    print(
        "\n========== EMBEDDING =========="
    )

    print(
        "Shape:",
        result["embedding"].shape
    )

    print(
        "\nFirst 10 values:"
    )

    print(
        result["embedding"][:10]
    )