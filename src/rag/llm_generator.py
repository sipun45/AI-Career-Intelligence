import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
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
# Load Knowledge
# ==========================================

def load_knowledge():

    with open(
        KNOWLEDGE_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return file.read()


# ==========================================
# Create Chunks
# ==========================================

def create_chunks(text):

    words = text.split()

    chunks = []

    for i in range(0, len(words), CHUNK_SIZE):

        chunk = " ".join(
            words[i:i + CHUNK_SIZE]
        )

        if chunk.strip():
            chunks.append(chunk)

    return chunks


# ==========================================
# Build Vector Index
# ==========================================

def build_index(chunks, model):

    embeddings = model.encode(chunks)

    embeddings = np.array(
        embeddings
    ).astype("float32")

    index = faiss.IndexFlatL2(
        embeddings.shape[1]
    )

    index.add(embeddings)

    return index


# ==========================================
# Retrieve Context
# ==========================================

def retrieve_context(
    question,
    model,
    index,
    chunks,
    k=3
):

    question_embedding = model.encode(
        [question]
    )

    question_embedding = np.array(
        question_embedding
    ).astype("float32")

    distances, indices = index.search(
        question_embedding,
        k
    )

    contexts = []

    for idx in indices[0]:

        if idx < len(chunks):

            contexts.append(
                chunks[idx]
            )

    return "\n\n".join(contexts)


# ==========================================
# Generate LLM Answer
# ==========================================

def generate_answer(
    question,
    context
):

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:

        return (
            "OpenAI API key is not configured.\n\n"
            "Retrieved career information:\n\n"
            + context
        )

    try:

        client = OpenAI(
            api_key=api_key
        )

        response = client.responses.create(

            model="gpt-4.1-mini",

            input=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI career assistant. "
                        "Answer using the provided career "
                        "knowledge. If the context does not "
                        "contain enough information, clearly "
                        "say so."
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

        return response.output_text

    except Exception as error:

        print(
            f"\nLLM error: {error}"
        )

        return (
            "The LLM could not generate an answer "
            "right now.\n\n"
            "Retrieved career information:\n\n"
            + context
        )


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":

    print("Loading embedding model...")

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    knowledge = load_knowledge()

    chunks = create_chunks(
        knowledge
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    index = build_index(
        chunks,
        model
    )

    print("FAISS index ready.")

    question = input(
        "\nAsk a career question: "
    )

    context = retrieve_context(
        question,
        model,
        index,
        chunks
    )

    print("\n===================================")
    print("GENERATING AI CAREER ANSWER")
    print("===================================")

    answer = generate_answer(
        question,
        context
    )

    print("\n" + answer)