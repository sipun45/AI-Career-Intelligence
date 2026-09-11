# 🚀 AI Career Intelligence & Job Recommendation System

An end-to-end AI/ML career intelligence platform that analyzes a candidate's skills, education, and experience to predict suitable job roles, estimate salary, recommend jobs, identify skill gaps, generate learning roadmaps, and provide AI-powered career assistance.

---

## 🎯 Project Overview

The AI Career Intelligence system combines traditional Machine Learning, Deep Learning, NLP, Transformers, RAG, and AI Agents into a single career recommendation platform.

### The system can:

- Predict expected salary
- Predict suitable job roles
- Recommend relevant jobs
- Calculate job matching scores
- Analyze candidate skills
- Identify skill gaps
- Generate personalized learning roadmaps
- Perform semantic resume/job matching
- Parse resumes
- Extract skills from resumes
- Provide RAG-based career assistance
- Run an AI Career Agent
- Monitor ML model performance
- Track experiments using MLflow

---

# 🏗️ System Architecture

```text
                         USER
                           │
                           ▼
                  ┌─────────────────┐
                  │ React Frontend  │
                  │   Port 5173     │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │   FastAPI APIs  │
                  └────────┬────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
   ML Models           NLP System       AI Systems
        │                  │                  │
        ▼                  ▼                  ▼
 Regression          Transformers          RAG
 Classification      Embeddings            Career Agent
 Clustering          Resume Parser         LLM
 Deep Learning       Skill Extraction
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ▼
                  Recommendations
                           │
                           ▼
                    Career Insights