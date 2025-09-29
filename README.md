# resume-reviewer

This project helps analyze a **resume against a job description** using **HuggingFace embeddings** (for similarity scoring) and **Groq LLM** (for structured review generation).

It works in **two ways**:

1. **Core logic only** – for quick understanding, refer to **`main.py`**.
2. **Full Web App** – for an interactive interface, refer to **`app.py`** (Flask backend) and **`templates/index.html`** (frontend).

---

## 🚀 Features

* Extracts resume text from PDF.
* Computes **Resume ↔ Job Description similarity score**.
* Generates a **detailed AI-powered review**:

  * Strengths
  * Weaknesses / Gaps
  * Hiring Verdict
* Web interface with **resume upload, job description input, score display, review, and downloadable TXT report**.

---

## 📂 Project Structure

```
resume_reviewer/
│-- app.py                # Flask web app
│-- main.py               # Core logic only (CLI usage)
│-- config.py             # API keys (HF + Groq)
│-- requirements.txt      # Python dependencies
│-- JD.txt                # Sample Job Description (Java Developer)
│-- templates/
│   └── index.html        # Frontend
│-- static/
│   └── style.css         # Optional styling
```

---

## 🔑 Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add API keys

Create `config.py` and include:

```python
HF_API_KEY = "your_huggingface_api_key"
GROQ_API_KEY = "your_groq_api_key"
```

### 3. Run Web App

```bash
python app.py
```

Go to → `http://127.0.0.1:5000/`

---

## 🧑‍💻 Usage

### CLI (Logic only)

```bash
python main.py
```

### Web App

* Upload your resume (`resume.pdf`)
* Paste the job description or use `JD.txt`
* Click **Analyze**
* Get similarity score + review
* Download TXT report

---

## 📘 Notes

* If you only want to **understand the core logic**, check **`main.py`**.
* If you want the **full webpage with interactivity**, use **`app.py`** + **`index.html`**.
* Modify **JD.txt** with your own job description to test different roles.

---
