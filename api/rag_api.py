import os

# ==========================================
# Memory Optimization
# ==========================================

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ==========================================
# Imports
# ==========================================

import numpy as np
import faiss

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv
from openai import OpenAI


# ==========================================
# Configuration
# ==========================================

load_dotenv()

KNOWLEDGE_PATH = "data/raw/career_knowledge.txt"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 500


# ==========================================
# FastAPI Application
# ==========================================

app = FastAPI(
    title="AI Career RAG API",
    version="1.0"
)


# ==========================================
# CORS
# ==========================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173"
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]
)


# ==========================================
# Request Model
# ==========================================

class CareerQuestion(BaseModel):

    question: str


# ==========================================
# Load Knowledge
# ==========================================

print("Loading career knowledge...")

with open(
    KNOWLEDGE_PATH,
    "r",
    encoding="utf-8"
) as file:

    knowledge = file.read()


print("Career knowledge loaded.")


# ==========================================
# Create Chunks
# ==========================================

words = knowledge.split()

chunks = []

for i in range(
    0,
    len(words),
    CHUNK_SIZE
):

    chunk = " ".join(
        words[i:i + CHUNK_SIZE]
    )

    if chunk.strip():

        chunks.append(chunk)


print(
    f"Created {len(chunks)} knowledge chunks."
)


# ==========================================
# Load Embedding Model
# ==========================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model loaded.")


# ==========================================
# Create Embeddings
# ==========================================

print("Creating embeddings...")

embeddings = embedding_model.encode(
    chunks
)

embeddings = np.array(
    embeddings
).astype("float32")


print(
    f"Embedding shape: {embeddings.shape}"
)


# ==========================================
# Create FAISS Index
# ==========================================

index = faiss.IndexFlatL2(
    embeddings.shape[1]
)

index.add(
    embeddings
)


print(
    f"RAG index ready with {len(chunks)} chunks."
)


# ==========================================
# Retrieve Relevant Context
# ==========================================

def retrieve_context(
    question,
    k=3
):

    # --------------------------------------
    # Create question embedding
    # --------------------------------------

    question_embedding = embedding_model.encode(
        [question]
    )

    question_embedding = np.array(
        question_embedding
    ).astype("float32")


    # --------------------------------------
    # Prevent duplicate results
    # when fewer chunks exist
    # --------------------------------------

    k = min(
        k,
        len(chunks)
    )


    # --------------------------------------
    # Search FAISS
    # --------------------------------------

    distances, indices = index.search(
        question_embedding,
        k
    )


    # --------------------------------------
    # Collect contexts
    # --------------------------------------

    contexts = []

    for idx in indices[0]:

        if idx < len(chunks):

            contexts.append(
                chunks[idx]
            )


    # --------------------------------------
    # Return context
    # --------------------------------------

    return "\n\n".join(
        contexts
    )


# ==========================================
# Generate LLM Answer
# ==========================================

def generate_answer(
    question,
    context
):

    # --------------------------------------
    # Get OpenAI API key
    # --------------------------------------

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )


    # --------------------------------------
    # API key missing
    # --------------------------------------

    if not api_key:

        return {
            "answer": (
                "OpenAI API key is not configured."
            ),

            "context": context
        }


    try:

        # ----------------------------------
        # Create OpenAI client
        # ----------------------------------

        client = OpenAI(
            api_key=api_key
        )


        # ----------------------------------
        # Generate response
        # ----------------------------------

        response = client.responses.create(

            model="gpt-4.1-mini",

            input=[
                {
                    "role": "system",

                    "content": (
                        "You are an AI career assistant. "
                        "Answer the user's question using "
                        "the provided career context. "
                        "Give practical, clear, and useful "
                        "career advice. "
                        "Do not invent information that is "
                        "not supported by the context."
                    )
                },

                {
                    "role": "user",

                    "content": (
                        f"Career Context:\n\n"
                        f"{context}\n\n"
                        f"Question:\n{question}"
                    )
                }
            ]
        )


        # ----------------------------------
        # Return generated answer
        # ----------------------------------

        return {

            "answer": response.output_text,

            "context": context
        }


    except Exception as error:

        # ----------------------------------
        # Handle LLM errors
        # ----------------------------------

        print(
            f"LLM error: {error}"
        )


        return {

            "answer": (
                "The LLM could not generate "
                "an answer right now."
            ),

            "error": str(error),

            "context": context
        }


# ==========================================
# Health Check
# ==========================================

@app.get("/")
def root():

    return {

        "message":
        "AI Career RAG API is running",

        "chunks":
        len(chunks),

        "embedding_model":
        EMBEDDING_MODEL
    }


# ==========================================
# RAG Assistant Endpoint
# ==========================================

@app.post("/rag-assistant")
def rag_assistant(
    request: CareerQuestion
):

    # --------------------------------------
    # Retrieve relevant context
    # --------------------------------------

    context = retrieve_context(
        request.question
    )


    # --------------------------------------
    # Generate answer
    # --------------------------------------

    result = generate_answer(
        request.question,
        context
    )


    # --------------------------------------
    # Return response
    # --------------------------------------

    return {

        "question":
        request.question,

        **result
    }   