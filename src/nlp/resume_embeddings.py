import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from sentence_transformers import SentenceTransformer
import numpy as np

from text_preprocessor import (
    clean_text,
    extract_sections
)


# Load Transformer model
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def create_resume_embedding(resume_text: str):

    # Clean resume
    cleaned_text = clean_text(
        resume_text
    )

    # Extract sections
    sections = extract_sections(
        resume_text
    )

    # Create structured resume text
    structured_text = f"""
    Skills:
    {sections['skills']}

    Education:
    {sections['education']}

    Experience:
    {sections['experience']}

    Projects:
    {sections['projects']}
    """

    # Create embedding
    embedding = model.encode(
        structured_text
    )

    return {
        "cleaned_text": cleaned_text,
        "sections": sections,
        "embedding": np.array(
            embedding
        )
    }


if __name__ == "__main__":

    resume = """
    JOHN DOE

    SKILLS
    Python
    SQL
    Machine Learning
    Deep Learning

    EDUCATION
    Bachelor of Technology in Computer Science

    EXPERIENCE
    Python Developer with 2 years experience.

    PROJECTS
    AI Career Recommendation System
    """

    result = create_resume_embedding(
        resume
    )

    print("========== RESUME SECTIONS ==========")

    for section, content in result[
        "sections"
    ].items():

        print(f"\n{section.upper()}:")
        print(content)

    print(
        "\n========== EMBEDDING =========="
    )

    print(
        "Shape:",
        result["embedding"].shape
    )

    print(
        "First 10 values:"
    )

    print(
        result["embedding"][:10]
    )