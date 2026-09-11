import os

# Prevent excessive CPU/memory usage
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

from src.nlp.semantic_matcher import calculate_similarity


# ==========================================
# FastAPI Application
# ==========================================

app = FastAPI(
    title="AI Career Intelligence - Job Matching API",
    description="Hybrid AI Job Recommendation System",
    version="1.0"
)


# ==========================================
# CORS
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:5174",
        "http://localhost:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# Load Dataset
# ==========================================

jobs_df = pd.read_csv(
    "data/raw/jobs.csv"
)


# ==========================================
# Request Model
# ==========================================

class JobMatchingInput(BaseModel):

    resume_text: str

    skills: list[str] = []

    experience: int = 0

    education: str = "Bachelor"

    target_role: str = "ML Engineer"


# ==========================================
# Home Route
# ==========================================

@app.get("/")
def home():

    return {
        "message": "Hybrid Job Matching API is running"
    }


# ==========================================
# Job Matching
# ==========================================

@app.post("/match-jobs")
def match_jobs(data: JobMatchingInput):

    # ======================================
    # 1. Create Job Text
    # ======================================

    job_texts = []

    for _, job in jobs_df.iterrows():

        job_text = f"""
        Job Title: {job['title']}
        Experience: {job['experience']} years
        Education: {job['education']}

        Python: {job['python']}
        Java: {job['java']}
        SQL: {job['sql']}
        Machine Learning: {job['ml']}
        Deep Learning: {job['deep_learning']}
        Cloud: {job['cloud']}
        """

        job_texts.append(job_text)


    # ======================================
    # 2. Semantic Similarity Score
    # ======================================

    semantic_scores = []

    for job_text in job_texts:

        score = calculate_similarity(
            data.resume_text,
            job_text
        )

        semantic_scores.append(score)


    # ======================================
    # 3. Prepare Results
    # ======================================

    results = jobs_df[
    [
        "job_id",
        "title",
        "experience",
        "education",
        "salary",
        "python",
        "java",
        "sql",
        "ml",
        "deep_learning",
        "cloud",
    ]
].copy()


    


    # Add semantic score

    results["semantic_score"] = semantic_scores


    # ======================================
    # 4. Experience Score
    # ======================================

    def calculate_experience_score(
        job_experience
    ):

        difference = abs(
            data.experience - job_experience
        )

        score = max(
            0,
            100 - (difference * 20)
        )

        return score


    results["experience_score"] = (
        results["experience"]
        .apply(
            calculate_experience_score
        )
    )


    # ======================================
    # 5. Education Score
    # ======================================

    results["education_score"] = (
        results["education"]
        .apply(
            lambda education:

            100
            if education.lower()
            == data.education.lower()

            else 50
        )
    )


    # ======================================
    # 6. Candidate Skills
    # ======================================

    candidate_skills = {
        skill.lower().strip()
        for skill in data.skills
    }


    # ======================================
    # 7. Skill Match Score
    # ======================================

    def calculate_skill_score(row):

        job_skills = []


        if row["python"] == 1:

            job_skills.append(
                "python"
            )


        if row["java"] == 1:

            job_skills.append(
                "java"
            )


        if row["sql"] == 1:

            job_skills.append(
                "sql"
            )


        if row["ml"] == 1:

            job_skills.append(
                "machine learning"
            )


        if row["deep_learning"] == 1:

            job_skills.append(
                "deep learning"
            )


        if row["cloud"] == 1:

            job_skills.append(
                "cloud"
            )


        # No required skills

        if not job_skills:

            return 0


        # Count matching skills

        matched = sum(
            skill in candidate_skills
            for skill in job_skills
        )


        return (
            matched
            / len(job_skills)
        ) * 100


    results["skill_score"] = (
        results.apply(
            calculate_skill_score,
            axis=1
        )
    )


    # ======================================
    # 8. Hybrid Match Score
    # ======================================

    results["match_score"] = (

        results["semantic_score"]
        * 0.50

        +

        results["skill_score"]
        * 0.20

        +

        results["experience_score"]
        * 0.20

        +

        results["education_score"]
        * 0.10
    )


    # ======================================
    # 9. Sort by Match Score
    # ======================================

    results = results.sort_values(
        "match_score",
        ascending=False
    )


    # ======================================
    # 10. Remove Duplicate Roles
    # ======================================

    results = results.drop_duplicates(
        subset=["title"],
        keep="first"
    )


    # ======================================
    # 11. Get Top 5 Jobs
    # ======================================

    results = results.head(5)


    # ======================================
    # 12. Return Recommendations
    # ======================================

    recommendations = []


    for _, row in results.iterrows():

        recommendations.append({

            "job_id": int(
                row["job_id"]
            ),

            "title": row["title"],

            "experience": int(
                row["experience"]
            ),

            "education": row["education"],

            "salary": int(
                row["salary"]
            ),

            "semantic_score": round(
                float(
                    row["semantic_score"]
                ),
                2
            ),

            "skill_score": round(
                float(
                    row["skill_score"]
                ),
                2
            ),

            "experience_score": round(
                float(
                    row["experience_score"]
                ),
                2
            ),

            "education_score": round(
                float(
                    row["education_score"]
                ),
                2
            ),

            "match_score": round(
                float(
                    row["match_score"]
                ),
                2
            ),
            "explanation": {
    "semantic": (
        "Strong semantic similarity"
        if row["semantic_score"] >= 70
        else "Moderate semantic similarity"
    ),

    "skills": (
        "Strong skill match"
        if row["skill_score"] >= 80
        else "Partial skill match"
    ),

    "experience": (
        "Experience matches the role"
        if row["experience_score"] >= 80
        else "Experience differs from the role"
    ),

    "education": (
        "Education requirement matches"
        if row["education_score"] == 100
        else "Education partially matches"
    )
},
        })

    return {
        "recommendations": recommendations
    }

# ==========================================
# Job-Specific Skill Gap
# ==========================================

@app.post("/job-skill-gap")
def job_skill_gap(data: JobMatchingInput):

    candidate_skills = {
        skill.lower().strip()
        for skill in data.skills
    }

    # Job-specific required skills
    job_required_skills = {
        "ML Engineer": {
            "python": "Advanced Python",
            "sql": "SQL and database management",
            "machine learning": "Machine Learning",
            "deep learning": "Deep Learning with PyTorch",
            "cloud": "Cloud computing",
            "docker": "Docker and containerization",
            "mlops": "MLOps and deployment",
        },

        "Data Scientist": {
            "python": "Advanced Python and Data Science",
            "sql": "SQL and database management",
            "machine learning": "Machine Learning",
            "statistics": "Statistics and probability",
            "pandas": "Pandas and data manipulation",
            "visualization": "Data visualization",
            "deep learning": "Deep Learning fundamentals",
        },

        "Data Analyst": {
            "python": "Python for data analysis",
            "sql": "Advanced SQL",
            "excel": "Advanced Excel",
            "pandas": "Pandas and data manipulation",
            "visualization": "Data visualization",
            "statistics": "Statistics and analytics",
        },

        "AI Engineer": {
            "python": "Advanced Python",
            "machine learning": "Machine Learning",
            "deep learning": "Deep Learning with PyTorch",
            "nlp": "Natural Language Processing",
            "llm": "Large Language Models",
            "rag": "Retrieval Augmented Generation",
            "docker": "Docker and containerization",
            "mlops": "MLOps and deployment",
        }
    }

    target_role = data.target_role if hasattr(data, "target_role") else "ML Engineer"

    required_skills = job_required_skills.get(
        target_role,
        job_required_skills["ML Engineer"]
    )

    matched_skills = []
    missing_skills = []

    for skill, learning_topic in required_skills.items():

        if skill in candidate_skills:

            matched_skills.append(skill)

        else:

            missing_skills.append({
                "skill": skill,
                "learning": learning_topic
            })

    total_required = len(required_skills)

    match_percentage = (
        len(matched_skills) / total_required
    ) * 100

    gap_percentage = 100 - match_percentage

    return {
        "target_role": target_role,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "skill_match_percentage": round(
            match_percentage,
            2
        ),
        "skill_gap_percentage": round(
            gap_percentage,
            2
        )
    }
# ==========================================
# Recommended Job Skill Gap
# ==========================================

@app.post("/job-recommendation-skill-gap")
def recommendation_skill_gap(data: JobMatchingInput):

    candidate_skills = {
        skill.lower().strip()
        for skill in data.skills
    }

    job_required_skills = {
        "ML Engineer": {
            "python": "Advanced Python",
            "sql": "SQL and database management",
            "machine learning": "Machine Learning",
            "deep learning": "Deep Learning with PyTorch",
            "cloud": "Cloud computing",
            "docker": "Docker and containerization",
            "mlops": "MLOps and deployment",
        },

        "Data Scientist": {
            "python": "Advanced Python and Data Science",
            "sql": "SQL and database management",
            "machine learning": "Machine Learning",
            "statistics": "Statistics and probability",
            "pandas": "Pandas and data manipulation",
            "visualization": "Data visualization",
            "deep learning": "Deep Learning fundamentals",
        },

        "Data Analyst": {
            "python": "Python for data analysis",
            "sql": "Advanced SQL",
            "excel": "Advanced Excel",
            "pandas": "Pandas and data manipulation",
            "visualization": "Data visualization",
            "statistics": "Statistics and analytics",
        },

        "AI Engineer": {
            "python": "Advanced Python",
            "machine learning": "Machine Learning",
            "deep learning": "Deep Learning with PyTorch",
            "nlp": "Natural Language Processing",
            "llm": "Large Language Models",
            "rag": "Retrieval Augmented Generation",
            "docker": "Docker and containerization",
            "mlops": "MLOps and deployment",
        }
    }

    target_role = data.target_role

    required_skills = job_required_skills.get(
        target_role,
        job_required_skills["ML Engineer"]
    )

    matched_skills = []
    missing_skills = []

    for skill, learning_topic in required_skills.items():

        if skill in candidate_skills:

            matched_skills.append(skill)

        else:

            missing_skills.append({
                "skill": skill,
                "learning": learning_topic
            })

    total_required = len(required_skills)

    match_percentage = (
        len(matched_skills) / total_required
    ) * 100

    gap_percentage = 100 - match_percentage

    return {
        "target_role": target_role,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "skill_match_percentage": round(
            match_percentage,
            2
        ),
        "skill_gap_percentage": round(
            gap_percentage,
            2
        )
    }