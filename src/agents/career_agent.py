import os

# ==========================================
# Memory Optimization
# ==========================================

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


import pandas as pd


# ==========================================
# Configuration
# ==========================================

JOBS_PATH = "data/raw/jobs.csv"


# ==========================================
# Load Jobs
# ==========================================

def load_jobs():

    return pd.read_csv(
        JOBS_PATH
    )


# ==========================================
# Candidate Skill Analysis
# ==========================================

def analyze_candidate_skills(candidate):

    skills = [
        "python",
        "java",
        "sql",
        "ml",
        "deep_learning",
        "cloud"
    ]

    candidate_skills = []

    for skill in skills:

        if candidate.get(skill, 0) == 1:

            candidate_skills.append(
                skill
            )

    return candidate_skills


# ==========================================
# Job Matching
# ==========================================

def calculate_job_match(
    candidate,
    job
):

    skills = [
        "python",
        "java",
        "sql",
        "ml",
        "deep_learning",
        "cloud"
    ]

    # --------------------------------------
    # Skill Score
    # --------------------------------------

    matched = 0

    for skill in skills:

        if (
            candidate.get(skill, 0) == 1
            and job[skill] == 1
        ):

            matched += 1

    skill_score = (
        matched / len(skills)
    ) * 100


    # --------------------------------------
    # Experience Score
    # --------------------------------------

    candidate_exp = candidate.get(
        "experience",
        0
    )

    job_exp = job["experience"]

    if candidate_exp >= job_exp:

        experience_score = 100

    else:

        experience_score = (
            candidate_exp / max(job_exp, 1)
        ) * 100


    # --------------------------------------
    # Education Score
    # --------------------------------------

    candidate_education = candidate.get(
        "education_encoded",
        0
    )

    job_education = (
        1
        if job["education"] == "Master"
        else 0
    )

    if candidate_education >= job_education:

        education_score = 100

    else:

        education_score = 50


    # --------------------------------------
    # Final Match Score
    # --------------------------------------

    final_score = (

        skill_score * 0.60

        + experience_score * 0.25

        + education_score * 0.15

    )

    return round(
        final_score,
        2
    )


# ==========================================
# Recommend Jobs
# ==========================================

def recommend_jobs(
    candidate,
    top_n=5
):

    jobs = load_jobs()

    recommendations = []


    # --------------------------------------
    # Calculate score for every job
    # --------------------------------------

    for _, job in jobs.iterrows():

        score = calculate_job_match(
            candidate,
            job
        )

        recommendations.append({

            "job_id":
            int(job["job_id"]),

            "title":
            job["title"],

            "experience":
            int(job["experience"]),

            "education":
            job["education"],

            "salary":
            int(job["salary"]),

            "match_score":
            score
        })


    # --------------------------------------
    # Sort by match score
    # --------------------------------------

    recommendations.sort(
        key=lambda x: x["match_score"],
        reverse=True
    )


    # --------------------------------------
    # Remove duplicate job titles
    # --------------------------------------

    unique_recommendations = []

    seen_roles = set()


    for recommendation in recommendations:

        role = recommendation["title"]


        if role not in seen_roles:

            seen_roles.add(
                role
            )

            unique_recommendations.append(
                recommendation
            )


        # Stop after required number
        # of unique roles

        if len(unique_recommendations) >= top_n:

            break


    return unique_recommendations


# ==========================================
# Skill Gap Analysis
# ==========================================

def calculate_skill_gap(
    candidate,
    job
):

    skills = [
        "python",
        "java",
        "sql",
        "ml",
        "deep_learning",
        "cloud"
    ]

    missing = []

    matched = []


    # --------------------------------------
    # Compare candidate with job
    # --------------------------------------

    for skill in skills:

        candidate_has_skill = (
            candidate.get(skill, 0) == 1
        )

        job_requires_skill = (
            job[skill] == 1
        )


        if job_requires_skill:

            if candidate_has_skill:

                matched.append(
                    skill
                )

            else:

                missing.append(
                    skill
                )


    return {

        "matched_skills":
        matched,

        "missing_skills":
        missing
    }


# ==========================================
# Learning Recommendations
# ==========================================

def learning_recommendations(
    missing_skills
):

    learning_map = {

        "python":
        "Learn Python programming, NumPy, Pandas and data structures.",

        "java":
        "Learn Core Java, OOP, collections and backend development.",

        "sql":
        "Learn SQL queries, joins, aggregation and database design.",

        "ml":
        "Learn supervised learning, unsupervised learning and model evaluation.",

        "deep_learning":
        "Learn neural networks, CNNs, RNNs, Transformers and PyTorch.",

        "cloud":
        "Learn AWS, Azure or GCP fundamentals and cloud deployment."
    }


    recommendations = []


    for skill in missing_skills:

        if skill in learning_map:

            recommendations.append({

                "skill":
                skill,

                "learning":
                learning_map[skill]
            })


    return recommendations


# ==========================================
# Career Agent
# ==========================================

def career_agent(
    candidate
):

    # ======================================
    # 1. Analyze Candidate Skills
    # ======================================

    candidate_skills = analyze_candidate_skills(
        candidate
    )


    # ======================================
    # 2. Recommend Jobs
    # ======================================

    recommendations = recommend_jobs(
        candidate,
        top_n=5
    )


    # Load jobs

    jobs = load_jobs()


    # ======================================
    # 3. Find Best Job
    # ======================================

    if recommendations:

        best_job_id = recommendations[0][
            "job_id"
        ]


        matching_jobs = jobs[
            jobs["job_id"] == best_job_id
        ]


        if not matching_jobs.empty:

            best_job = matching_jobs.iloc[0]


            # ==================================
            # 4. Skill Gap
            # ==================================

            skill_gap = calculate_skill_gap(
                candidate,
                best_job
            )

        else:

            skill_gap = {

                "matched_skills": [],

                "missing_skills": []
            }

    else:

        skill_gap = {

            "matched_skills": [],

            "missing_skills": []
        }


    # ======================================
    # 5. Learning Plan
    # ======================================

    learning = learning_recommendations(
        skill_gap[
            "missing_skills"
        ]
    )


    # ======================================
    # 6. Best Role
    # ======================================

    if recommendations:

        best_role = recommendations[0][
            "title"
        ]

        best_score = recommendations[0][
            "match_score"
        ]

    else:

        best_role = (
            "No suitable role found"
        )

        best_score = 0


    # ======================================
    # 7. Career Summary
    # ======================================

    career_summary = (

        f"Based on your skills and experience, "
        f"your best matching career role is "
        f"{best_role} with a match score of "
        f"{best_score}%."

    )


    # ======================================
    # 8. Return Agent Result
    # ======================================

    return {

        "candidate_skills":
        candidate_skills,

        "recommended_role":
        best_role,

        "match_score":
        best_score,

        "career_summary":
        career_summary,

        "job_recommendations":
        recommendations,

        "skill_gap":
        skill_gap,

        "learning_plan":
        learning
    }


# ==========================================
# Test Agent
# ==========================================

if __name__ == "__main__":


    # ======================================
    # Test Candidate
    # ======================================

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


    # ======================================
    # Run Career Agent
    # ======================================

    result = career_agent(
        candidate
    )


    # ======================================
    # Display Result
    # ======================================

    print(
        "\n==================================="
    )

    print(
        "        AI CAREER AGENT"
    )

    print(
        "==================================="
    )


    # --------------------------------------
    # Candidate Skills
    # --------------------------------------

    print(
        "\nCandidate Skills:"
    )

    print(
        result[
            "candidate_skills"
        ]
    )


    # --------------------------------------
    # Recommended Role
    # --------------------------------------

    print(
        "\nRecommended Role:"
    )

    print(
        result[
            "recommended_role"
        ]
    )


    # --------------------------------------
    # Match Score
    # --------------------------------------

    print(
        "\nMatch Score:"
    )

    print(
        f"{result['match_score']}%"
    )


    # --------------------------------------
    # Career Summary
    # --------------------------------------

    print(
        "\nCareer Summary:"
    )

    print(
        result[
            "career_summary"
        ]
    )


    # --------------------------------------
    # Job Recommendations
    # --------------------------------------

    print(
        "\nJob Recommendations:"
    )


    for job in result[
        "job_recommendations"
    ]:

        print(

            f"{job['title']} "
            f"- {job['match_score']}% "
            f"- Salary: ${job['salary']}"

        )


    # --------------------------------------
    # Skill Gap
    # --------------------------------------

    print(
        "\nSkill Gap:"
    )

    print(
        result[
            "skill_gap"
        ]
    )


    # --------------------------------------
    # Learning Plan
    # --------------------------------------

    print(
        "\nLearning Plan:"
    )


    for item in result[
        "learning_plan"
    ]:

        print(

            f"{item['skill']} → "
            f"{item['learning']}"

        )


    print(
        "\n==================================="
    )

    print(
        "       AGENT EXECUTION COMPLETE"
    )

    print(
        "==================================="
    )