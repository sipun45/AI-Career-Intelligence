import os

# ==========================================
# Memory Optimization
# ==========================================

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ==========================================
# Import Agent + RAG
# ==========================================

from src.agents.rag_career_agent import (
    generate_career_guidance
)


# ==========================================
# FastAPI Application
# ==========================================

app = FastAPI(
    title="AI Career Agent API",
    description="Career Agent + RAG API",
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
        "http://localhost:5174"
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]
)


# ==========================================
# Request Model
# ==========================================

class CandidateRequest(BaseModel):

    experience: int

    education_encoded: int

    python: int

    java: int

    sql: int

    ml: int

    deep_learning: int

    cloud: int


# ==========================================
# Root Endpoint
# ==========================================

@app.get("/")
def home():

    return {

        "message":
        "AI Career Agent API is running",

        "status":
        "success"
    }


# ==========================================
# Career Agent Endpoint
# ==========================================

@app.post("/career-agent")
def run_career_agent(
    candidate: CandidateRequest
):

    try:

        # Convert request to dictionary

        candidate_data = (
            candidate.model_dump()
        )


        # Run Career Agent + RAG

        result = generate_career_guidance(
            candidate_data
        )


        return {

            "success":
            True,

            "result":
            result
        }


    except Exception as error:

        return {

            "success":
            False,

            "message":
            "Career Agent failed",

            "error":
            str(error)
        }


# ==========================================
# Run Information
# ==========================================

@app.get("/health")
def health():

    return {

        "status":
        "healthy",

        "service":
        "career-agent"
    }