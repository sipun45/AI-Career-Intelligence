import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader

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


@app.get("/")
def home():
    return {
        "message": "Resume API is running"
    }


@app.post("/extract-resume")
async def extract_resume(file: UploadFile = File(...)):

    if file.content_type != "application/pdf":
        return {
            "error": "Please upload a PDF file"
        }

    contents = await file.read()

    temp_file = "temp_resume.pdf"

    with open(temp_file, "wb") as f:
        f.write(contents)

    reader = PdfReader(temp_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    os.remove(temp_file)

    return {
        "filename": file.filename,
        "text": text
    }