import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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

print("Loading semantic model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Semantic model loaded successfully!")


class SemanticMatchInput(BaseModel):
    resume_text: str
    job_description: str


@app.get("/")
def home():
    return {
        "message": "Semantic Matching API is running"
    }


@app.post("/semantic-match")
def semantic_match(data: SemanticMatchInput):

    resume_embedding = model.encode(
        [data.resume_text]
    )

    job_embedding = model.encode(
        [data.job_description]
    )

    similarity = cosine_similarity(
        resume_embedding,
        job_embedding
    )[0][0]

    match_percentage = similarity * 100

    return {
        "match_percentage": round(
            float(match_percentage),
            2
        )
    }