@echo off

cd /d C:\Users\Sipun\Desktop\ML-Career-Intelligence

call .venv\Scripts\activate

echo Starting Main ML API - 8000
start "API 8000" cmd /k "python -m uvicorn api.main:app --port 8000"

echo Starting Semantic API - 8001
start "API 8001" cmd /k "python -m uvicorn src.nlp.semantic_api:app --port 8001"

echo Starting Resume API - 8002
start "API 8002" cmd /k "python -m uvicorn src.nlp.resume_api:app --port 8002"

echo Starting Skill Extraction API - 8003
start "API 8003" cmd /k "python -m uvicorn src.nlp.skill_extraction_api:app --port 8003"

echo Starting Learning Roadmap API - 8004
start "API 8004" cmd /k "python -m uvicorn src.nlp.learning_api:app --port 8004"

echo Starting Job Matching API - 8005
start "API 8005" cmd /k "python -m uvicorn src.nlp.job_matching_api:app --port 8005"

echo Starting Deep Learning API - 8006
start "API 8006" cmd /k "python -m uvicorn api.deep_learning_api:app --port 8006"

echo Starting RAG API - 8007
start "API 8007" cmd /k "python -m uvicorn api.rag_api:app --port 8007"

echo Starting Career Agent API - 8008
start "API 8008" cmd /k "python -m uvicorn api.career_agent_api:app --port 8008"

echo Starting React Frontend - 5173
start "Frontend 5173" cmd /k "cd /d C:\Users\Sipun\Desktop\ML-Career-Intelligence\frontend && npm run dev"

echo.
echo ========================================
echo All services are starting...
echo ========================================
pause