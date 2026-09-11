from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


app = FastAPI(
    title="Personalized Learning Roadmap API"
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
# Learning Paths
# ==========================================

learning_paths = {

    "ML Engineer": {

        "python":
            "Advanced Python and programming",

        "sql":
            "SQL and database optimization",

        "machine learning":
            "Supervised and unsupervised machine learning",

        "deep learning":
            "Neural networks and PyTorch",

        "cloud":
            "AWS or Azure cloud fundamentals",

        "docker":
            "Docker and containerization",

        "mlops":
            "ML pipelines, deployment and monitoring",
    },


    "Data Scientist": {

        "python":
            "Advanced Python and data science",

        "sql":
            "SQL and database management",

        "machine learning":
            "Machine learning algorithms",

        "statistics":
            "Statistics and probability",

        "pandas":
            "Pandas and data manipulation",

        "visualization":
            "Data visualization with Matplotlib and Seaborn",

        "deep learning":
            "Deep learning fundamentals",
    },


    "Data Analyst": {

        "python":
            "Python for data analysis",

        "sql":
            "Advanced SQL",

        "excel":
            "Advanced Excel",

        "pandas":
            "Pandas and data manipulation",

        "visualization":
            "Data visualization",

        "statistics":
            "Statistics and analytics",
    },


    "AI Engineer": {

        "python":
            "Advanced Python",

        "machine learning":
            "Machine learning",

        "deep learning":
            "Deep learning with PyTorch",

        "nlp":
            "Natural Language Processing",

        "transformers":
            "Transformers and attention mechanisms",

        "llm":
            "Large Language Models",

        "rag":
            "Retrieval Augmented Generation",

        "docker":
            "Docker and containerization",

        "mlops":
            "MLOps and model deployment",
    }
}


# ==========================================
# Request Model
# ==========================================

class LearningInput(BaseModel):

    skills: list[str] = []

    target_role: str


# ==========================================
# Roadmap Endpoint
# ==========================================

@app.post("/learning-roadmap")
def learning_roadmap(data: LearningInput):

    # ======================================
    # 1. Get Target Role
    # ======================================

    target_role = data.target_role.strip()


    # ======================================
    # 2. Check Role
    # ======================================

    if target_role not in learning_paths:

        return {
            "target_role": target_role,
            "roadmap": [],
            "message": "Target role not found"
        }


    # ======================================
    # 3. Normalize Candidate Skills
    # ======================================

    candidate_skills = {
        skill.lower().strip()
        for skill in data.skills
    }


    # ======================================
    # 4. Get Required Skills
    # ======================================

    required_skills = learning_paths[
        target_role
    ]


    # ======================================
    # 5. Calculate Skill Statistics
    # ======================================

    total_required_skills = len(
        required_skills
    )

    current_skill_count = 0


    for skill in required_skills:

        if skill.lower() in candidate_skills:

            current_skill_count += 1


    # ======================================
    # 6. Calculate Percentages
    # ======================================

    skill_gap_percentage = (
        (
            total_required_skills
            - current_skill_count
        )
        / total_required_skills
    ) * 100


    skill_match_percentage = (
        current_skill_count
        / total_required_skills
    ) * 100


    # ======================================
    # 7. Find Missing Skills
    # ======================================

    roadmap = []


    for skill, learning_topic in required_skills.items():

        if skill.lower() not in candidate_skills:

            roadmap.append({

                "skill": skill,

                "topic": learning_topic
            })


    # ======================================
    # 8. Return Response
    # ======================================

    return {

        "target_role": target_role,

        "current_skills": list(
            candidate_skills
        ),

        "total_required_skills":
            total_required_skills,

        "matched_skills":
            current_skill_count,

        "skill_match_percentage":
            round(
                skill_match_percentage,
                2
            ),

        "skill_gap_percentage":
            round(
                skill_gap_percentage,
                2
            ),

        "missing_skills": [
            item["skill"]
            for item in roadmap
        ],

        "roadmap":
            roadmap
    }