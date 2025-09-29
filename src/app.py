from flask import Flask, render_template, request
import pdfplumber
import numpy as np
from numpy import dot
from numpy.linalg import norm
from huggingface_hub import InferenceClient
from groq import Groq
from config import HF_API_KEY, GROQ_API_KEY

app = Flask(__name__)

# ---------------------------
# Utility: Load Resume Text
# ---------------------------
def load_resume(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

# ---------------------------
# Embeddings
# ---------------------------
def compute_embeddings(texts, hf_client):
    embeddings = hf_client.feature_extraction(
        texts,
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    return np.array(embeddings, dtype="float32")

def compute_similarity(resume_text, job_desc, hf_client):
    v1, v2 = compute_embeddings([resume_text, job_desc], hf_client)
    return dot(v1, v2) / (norm(v1) * norm(v2))

# ---------------------------
# Review with Groq
# ---------------------------
def generate_review(groq_client, resume_text, job_desc, score):
    prompt = f"""
You are an expert career coach and ATS optimization consultant.

Job Description:
{job_desc}

Resume:
{resume_text}

Resume-Job Match Score: {score:.2f}

Provide a detailed review with:
1. Strengths of this resume for the given job.
2. Weaknesses or gaps compared to the job description.
3. Overall verdict on hiring potential.
"""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    return response.choices[0].message.content.strip()

# ---------------------------
# Routes
# ---------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        resume_file = request.files["resume"]
        job_desc = request.form["job_desc"]

        if not resume_file or not job_desc:
            return render_template("index.html", error="Please upload a resume and enter job description.")

        # Initialize clients
        hf_client = InferenceClient(token=HF_API_KEY)
        groq_client = Groq(api_key=GROQ_API_KEY)

        # Process resume
        resume_text = load_resume(resume_file)
        score = compute_similarity(resume_text, job_desc, hf_client)
        review = generate_review(groq_client, resume_text, job_desc, score)

        return render_template("index.html", score=f"{score:.2f}", review=review)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
