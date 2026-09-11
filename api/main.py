import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5174",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained ML models
model = joblib.load("models/salary_model.pkl")
classifier = joblib.load("models/job_role_model.pkl")
# Load jobs dataset
jobs_df = pd.read_csv("data/raw/jobs.csv")

# Load career knowledge
with open(
    "data/raw/career_knowledge.txt",
    "r",
    encoding="utf-8"
) as file:
    career_text = file.read()

recommendation_features = [
    "python",
    "java",
    "sql",
    "ml",
    "deep_learning",
    "cloud"
]

job_vectors = jobs_df[recommendation_features]
# Load sentence transformer

# Career role skill requirements
career_skills = {
    "ML Engineer": [
        "python",
        "sql",
        "machine learning",
        "deep learning",
        "cloud",
        "docker",
        "mlops"
    ],

    "Data Scientist": [
        "python",
        "sql",
        "machine learning",
        "statistics",
        "pandas",
        "visualization"
    ],

    "Data Analyst": [
        "sql",
        "excel",
        "python",
        "statistics",
        "visualization"
    ],

    "AI Engineer": [
        "python",
        "sql",
        "machine learning",
        "deep learning",
        "cloud",
        "llm",
        "rag"
    ]
}


class SalaryInput(BaseModel):
    experience: int
    education_encoded: int
    python: int
    java: int
    sql: int
    ml: int
    deep_learning: int
    cloud: int
    total_skills: int


class SkillGapInput(BaseModel):
    skills: list[str]
    target_role: str


class JobRecommendationInput(BaseModel):
    python: int
    java: int
    sql: int
    ml: int
    deep_learning: int
    cloud: int

class CareerQuestionInput(BaseModel):
    question: str


class SemanticMatchInput(BaseModel):
    resume_text: str
    job_description: str


@app.get("/")
def home():
    return {
        "message": "Career Intelligence API is running"
    }


@app.post("/predict-salary")
def predict_salary(data: SalaryInput):

    features = [[
        data.experience,
        data.education_encoded,
        data.python,
        data.java,
        data.sql,
        data.ml,
        data.deep_learning,
        data.cloud,
        data.total_skills
    ]]

    prediction = model.predict(features)[0]

    return {
        "predicted_salary": round(float(prediction), 2)
    }

@app.post("/predict-role")
def predict_role(data: SalaryInput):

    features = [[
        data.experience,
        data.education_encoded,
        data.python,
        data.java,
        data.sql,
        data.ml,
        data.deep_learning,
        data.cloud,
        data.total_skills
    ]]

    prediction = classifier.predict(features)[0]

    return {
        "predicted_role": prediction
    }

@app.post("/analyze-candidate")
def analyze_candidate(data: SalaryInput):

    features = [[
        data.experience,
        data.education_encoded,
        data.python,
        data.java,
        data.sql,
        data.ml,
        data.deep_learning,
        data.cloud,
        data.total_skills
    ]]

    # Salary prediction
    salary_prediction = model.predict(features)[0]

    # Job role prediction
    role_prediction = classifier.predict(features)[0]

    # Prediction probability
    probabilities = classifier.predict_proba(features)[0]
    classes = classifier.classes_

    # Find top 3 recommended roles
    top_indices = probabilities.argsort()[-3:][::-1]

    recommendations = []

    for i in top_indices:
        recommendations.append({
            "role": classes[i],
            "probability": round(float(probabilities[i] * 100), 2)
        })

    return {
        "predicted_role": role_prediction,
        "predicted_salary": round(float(salary_prediction), 2),
        "top_roles": recommendations
    }

@app.post("/skill-gap")
def skill_gap(data: SkillGapInput):

    target_role = data.target_role

    # Check whether role exists
    if target_role not in career_skills:
        return {
            "error": "Role not found",
            "available_roles": list(career_skills.keys())
        }

    # Required skills for target role
    required_skills = set(career_skills[target_role])

    # Candidate skills
    candidate_skills = set(
        skill.lower() for skill in data.skills
    )

    # Find matched and missing skills
    matched_skills = required_skills.intersection(candidate_skills)
    missing_skills = required_skills.difference(candidate_skills)

    # Calculate percentage
    match_percentage = (
        len(matched_skills) / len(required_skills)
    ) * 100

    return {
        "target_role": target_role,
        "matched_skills": sorted(list(matched_skills)),
        "missing_skills": sorted(list(missing_skills)),
        "match_percentage": round(match_percentage, 2)
    }    


@app.post("/recommend-jobs")
def recommend_jobs(data: JobRecommendationInput):

    candidate = [[
        data.python,
        data.java,
        data.sql,
        data.ml,
        data.deep_learning,
        data.cloud
    ]]

    # Calculate similarity
    similarities = cosine_similarity(
        candidate,
        job_vectors
    )[0]

    # Create result
    recommendations = jobs_df[
        ["job_id", "title", "salary"]
    ].copy()

    recommendations["match_score"] = similarities * 100

    # Sort by highest match
    recommendations = recommendations.sort_values(
        "match_score",
        ascending=False
    )

    # Return top 5
    top_jobs = recommendations.head(5)

    return {
        "recommendations": top_jobs.to_dict(
            orient="records"
        )
    }


@app.post("/career-assistant")
def career_assistant(data: CareerQuestionInput):

    question = data.question.lower()

    # Split knowledge into paragraphs
    paragraphs = [
        p.strip()
        for p in career_text.split("\n\n")
        if p.strip()
    ]

    # Find paragraphs containing words from question
    question_words = set(question.split())

    scored_paragraphs = []

    for paragraph in paragraphs:

        paragraph_words = set(
            paragraph.lower().split()
        )

        score = len(
            question_words.intersection(paragraph_words)
        )

        scored_paragraphs.append(
            (score, paragraph)
        )

    # Sort by relevance
    scored_paragraphs.sort(
        key=lambda x: x[0],
        reverse=True
    )

    # Get top 3
    top_results = [
        paragraph
        for score, paragraph in scored_paragraphs[:3]
        if score > 0
    ]

    if not top_results:
        return {
            "question": data.question,
            "answer": "I do not have enough information in my career knowledge base."
        }

    return {
        "question": data.question,
        "answer": "\n\n".join(top_results)
    }