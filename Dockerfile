FROM python:3.11-slim

WORKDIR /app

COPY docker-requirements.txt .

RUN pip install --no-cache-dir -r docker-requirements.txt

COPY api ./api
COPY src ./src
COPY data ./data
COPY models ./models

EXPOSE 8000

CMD ["uvicorn", "api.production_main:app", "--host", "0.0.0.0", "--port", "8000"]