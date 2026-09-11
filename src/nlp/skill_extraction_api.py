import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Skills that our system can recognize
SKILLS = [
    "python",
    "java",
    "sql",
    "machine learning",
    "deep learning",
    "cloud",
    "docker",
    "mlops",
    "statistics",
    "pandas",
    "visualization",
    "excel",
    "llm",
    "rag",
    "tensorflow",
    "pytorch",
    "scikit-learn",
    "nlp",
    "aws",
    "azure",
    "gcp",
    "git",
    "mongodb",
    "fastapi",
    "react",
    "node.js",
]


class SkillExtractionInput(BaseModel):
    resume_text: str


@app.get("/")
def home():
    return {
        "message": "Skill Extraction API is running"
    }


@app.post("/extract-skills")
def extract_skills(data: SkillExtractionInput):

    text = data.resume_text.lower()

    detected_skills = []

    for skill in SKILLS:

        pattern = r"\b" + re.escape(skill.lower()) + r"\b"

        if re.search(pattern, text):
            detected_skills.append(skill)

    return {
        "skills": sorted(detected_skills),
        "skill_count": len(detected_skills)
    }