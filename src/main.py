# Install the required libraries:
# pip install pdfplumber huggingface-hub groq

import os
import numpy as np
import pdfplumber
from numpy import dot
from numpy.linalg import norm
from huggingface_hub import InferenceClient
from groq import Groq

hf_api_key = "YOUR_HF_API_KEY"
groq_api_key = "YOUR_GROQ_API_KEY"

def load_resume(filepath):
    """
    Extracts text from a PDF resume and returns it as a single string.
    """
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def compute_embeddings(texts, hf_client):
    """
    Generates embeddings for a list of texts using a sentence-transformer model.
    """
    embeddings = hf_client.feature_extraction(
        texts,
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    return np.array(embeddings, dtype="float32")

def compute_similarity(resume_text, job_desc, hf_client):
    """
    Computes cosine similarity between resume and job description embeddings.
    """
    v1, v2 = compute_embeddings([resume_text, job_desc], hf_client)
    similarity = dot(v1, v2) / (norm(v1) * norm(v2))
    return similarity
