import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


# ==========================================
# Configuration
# ==========================================

KNOWLEDGE_PATH = "data/raw/career_knowledge.txt"

MODEL_NAME = "all-MiniLM-L6-v2"

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

        text = file.read()

    return text


# ==========================================
# Create Chunks
# ==========================================

def create_chunks(text):

    words = text.split()

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

    return chunks


# ==========================================
# Create Embeddings
# ==========================================

def create_embeddings(
    chunks,
    model
):

    embeddings = model.encode(
        chunks
    )

    embeddings = np.array(
        embeddings
    ).astype("float32")

    return embeddings


# ==========================================
# Build FAISS Index
# ==========================================

def build_index(
    embeddings
):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(
        embeddings
    )

    return index


# ==========================================
# Retrieve Relevant Documents
# ==========================================

def retrieve(
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

    results = []

    for distance, idx in zip(
        distances[0],
        indices[0]
    ):

        if idx < len(chunks):

            results.append({
                "chunk": chunks[idx],
                "distance": float(distance)
            })

    return results


# ==========================================
# Main Test
# ==========================================

if __name__ == "__main__":

    print("Loading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    text = load_knowledge()

    chunks = create_chunks(
        text
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    embeddings = create_embeddings(
        chunks,
        model
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    index = build_index(
        embeddings
    )

    print("FAISS index created.")

    question = input(
        "\nAsk a career question: "
    )

    results = retrieve(
        question,
        model,
        index,
        chunks
    )

    print("\n===================================")
    print("RETRIEVED CAREER INFORMATION")
    print("===================================")

    for i, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\nResult {i}"
        )

        print(
            result["chunk"]
        )

        print(
            f"Distance: {result['distance']:.4f}"
        )