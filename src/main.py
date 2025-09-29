# Install the required libraries before running:
# pip install -r requirements.txt

import numpy as np
import pdfplumber
from numpy import dot
from numpy.linalg import norm
from huggingface_hub import InferenceClient
from groq import Groq
from config import HF_API_KEY, GROQ_API_KEY

# ---------------------------
# Utility: Load Resume from PDF
# ---------------------------
def load_resume(filepath):
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

# ---------------------------
# Compute Embeddings
# ---------------------------
def compute_embeddings(texts, hf_client):
    embeddings = hf_client.feature_extraction(
        texts,
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    return np.array(embeddings, dtype="float32")

# ---------------------------
# Compute Cosine Similarity
# ---------------------------
def compute_similarity(resume_text, job_desc, hf_client):
    v1, v2 = compute_embeddings([resume_text, job_desc], hf_client)
    similarity = dot(v1, v2) / (norm(v1) * norm(v2))
    return similarity

# ---------------------------
# Generate Structured Review using Groq
# ---------------------------
def generate_review(groq_client, resume_text, job_desc, score):
    """
    Generate structured resume feedback using Groq LLM.
    """
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
# Main Flow
# ---------------------------
if __name__ == "__main__":
    # Initialize clients
    hf_client = InferenceClient(token=HF_API_KEY)
    groq_client = Groq(api_key=GROQ_API_KEY)

    # Load resume and job description
    resume_text = load_resume("resume.pdf")
    with open("JD.txt", "r") as f:
        job_desc = f.read().strip()

    # Compute Resume-Job similarity
    print("\nEvaluating resume vs job description...")
    score = compute_similarity(resume_text, job_desc, hf_client)
    print(f"Resume-Job Match Score: {score:.2f}")

    # Generate structured review
    print("\nGenerating resume review...")
    review = generate_review(groq_client, resume_text, job_desc, score)
    print("\nResume Review Report:\n")
    print(review)
