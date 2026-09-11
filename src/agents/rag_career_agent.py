import os

# ==========================================
# Memory Optimization
# ==========================================

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


import sys
import numpy as np

from sentence_transformers import SentenceTransformer
import faiss


# ==========================================
# Import Career Agent
# ==========================================

from src.agents.career_agent import career_agent
# ==========================================
# Configuration
# ==========================================

KNOWLEDGE_PATH = "data/raw/career_knowledge.txt"

MODEL_NAME = "all-MiniLM-L6-v2"


# ==========================================
# Load Career Knowledge
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
# Create Knowledge Chunks
# ==========================================

def create_chunks(
    text,
    chunk_size=500
):

    words = text.split()

    chunks = []

    for i in range(
        0,
        len(words),
        chunk_size
    ):

        chunk = " ".join(
            words[
                i:i + chunk_size
            ]
        )

        if chunk.strip():

            chunks.append(
                chunk
            )

    return chunks


# ==========================================
# Load Embedding Model
# ==========================================

print(
    "Loading career knowledge..."
)

knowledge = load_knowledge()

chunks = create_chunks(
    knowledge
)

print(
    f"Created {len(chunks)} knowledge chunks."
)


print(
    "Loading embedding model..."
)

embedding_model = SentenceTransformer(
    MODEL_NAME
)

print(
    "Embedding model loaded."
)


# ==========================================
# Create Embeddings
# ==========================================

print(
    "Creating embeddings..."
)

embeddings = embedding_model.encode(
    chunks
)

embeddings = np.array(
    embeddings
).astype("float32")


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
    "RAG index ready."
)


# ==========================================
# Retrieve Knowledge
# ==========================================

def retrieve_knowledge(
    question,
    k=3
):

    question_embedding = (
        embedding_model.encode(
            [question]
        )
    )

    question_embedding = np.array(
        question_embedding
    ).astype("float32")


    # Prevent requesting more
    # chunks than available

    k = min(
        k,
        len(chunks)
    )


    distances, indices = index.search(
        question_embedding,
        k
    )


    retrieved_chunks = []


    for i in indices[0]:

        retrieved_chunks.append(
            chunks[i]
        )


    return retrieved_chunks


# ==========================================
# Generate Career Guidance
# ==========================================

def generate_career_guidance(
    candidate
):

    # --------------------------------------
    # Run Career Agent
    # --------------------------------------

    agent_result = career_agent(
        candidate
    )


    # --------------------------------------
    # Build Career Question
    # --------------------------------------

    recommended_role = (
        agent_result[
            "recommended_role"
        ]
    )


    missing_skills = (
        agent_result[
            "skill_gap"
        ][
            "missing_skills"
        ]
    )


    question = (

        f"How can I become a "
        f"{recommended_role}? "

        f"What should I learn about "
        f"{', '.join(missing_skills)}?"

    )


    # --------------------------------------
    # Retrieve RAG Knowledge
    # --------------------------------------

    retrieved = retrieve_knowledge(
        question,
        k=3
    )


    # --------------------------------------
    # Return Complete Result
    # --------------------------------------

    return {

        "career_agent":
        agent_result,

        "rag_question":
        question,

        "knowledge":
        retrieved
    }


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":


    candidate = {

        "experience":
        3,

        "education_encoded":
        1,

        "python":
        1,

        "java":
        0,

        "sql":
        1,

        "ml":
        1,

        "deep_learning":
        0,

        "cloud":
        0
    }


    print(
        "\n======================================"
    )

    print(
        "       AI CAREER AGENT + RAG"
    )

    print(
        "======================================"
    )


    result = generate_career_guidance(
        candidate
    )


    # ======================================
    # Career Agent Result
    # ======================================

    agent_result = result[
        "career_agent"
    ]


    print(
        "\nRecommended Role:"
    )

    print(
        agent_result[
            "recommended_role"
        ]
    )


    print(
        "\nMatch Score:"
    )

    print(
        f"{agent_result['match_score']}%"
    )


    print(
        "\nSkill Gap:"
    )

    print(
        agent_result[
            "skill_gap"
        ]
    )


    print(
        "\nLearning Plan:"
    )


    for item in agent_result[
        "learning_plan"
    ]:

        print(

            f"{item['skill']} → "
            f"{item['learning']}"

        )


    # ======================================
    # RAG Question
    # ======================================

    print(
        "\nRAG Question:"
    )

    print(
        result[
            "rag_question"
        ]
    )


    # ======================================
    # Retrieved Knowledge
    # ======================================

    print(
        "\nRetrieved Career Knowledge:"
    )


    for i, knowledge in enumerate(
        result["knowledge"],
        start=1
    ):

        print(
            f"\n--- Knowledge {i} ---"
        )

        print(
            knowledge
        )


    print(
        "\n======================================"
    )

    print(
        "      AGENT + RAG COMPLETE"
    )

    print(
        "======================================"
    )