from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.main import app as main_app
from api.deep_learning_api import app as deep_learning_app
from api.rag_api import app as rag_app
from api.career_agent_api import app as career_agent_app

from src.nlp.job_matching_api import app as job_matching_app
from src.nlp.learning_api import app as learning_app
from src.nlp.resume_api import app as resume_app
from src.nlp.semantic_api import app as semantic_app
from src.nlp.skill_extraction_api import app as skill_extraction_app


app = FastAPI(
    title="AI Career Intelligence - Production API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def add_routes(source_app):
    for route in source_app.routes:
        if route.path == "/":
            continue

        if route.path == "/health":
            continue

        app.router.routes.append(route)


# Main API
add_routes(main_app)

# NLP APIs
add_routes(job_matching_app)
add_routes(learning_app)
add_routes(resume_app)
add_routes(semantic_app)
add_routes(skill_extraction_app)

# Advanced APIs
add_routes(deep_learning_app)
add_routes(rag_app)
add_routes(career_agent_app)


@app.get("/")
def root():
    return {
        "message": "AI Career Intelligence API",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }