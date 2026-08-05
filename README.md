# AI Resume Screening & Job Recommendation System

An AI-powered web application that automates candidate resume parsing, technical and soft skill extraction, profile summarization, and deterministic job-role matching using **Gemini API** and **OpenRouter** LLM providers.

---

## 🌟 Overview

The **AI Resume Screening & Job Recommendation System** enables recruiters, HR managers, and job seekers to upload resumes in **PDF** or **DOCX** format and instantly receive comprehensive, actionable insights. 

The application extracts raw document text, utilizes Gemini / OpenRouter LLM APIs for natural language analysis, and compares extracted technical skills against predefined industry job role criteria. It provides match percentage scores, highlights missing required and preferred skills, and recommends targeted learning paths.

---

## ✨ Features

- **Multi-Provider AI Architecture**: Primary fast **Gemini API** (`gemini-2.0-flash`, 1-3s analysis) with automatic fallback to **OpenRouter** (`openrouter/free`).
- **Exponential Backoff Retries**: Automatically retries transient provider errors and rate limits while avoiding retries on invalid API keys.
- **Document Parsing**: Supports PDF (`pdfplumber`) and Word (`python-docx`) parsing with input validation (up to 5 MB).
- **AI-Powered Skill Extraction**: Automatically identifies technical skills, soft skills, programming languages, frameworks, databases, and tools.
- **Implicit Skill Inference**: Automatically infers foundational web skills (`HTML`, `CSS`, `JavaScript`) when modern frameworks like `React` or `Node.js` are present.
- **Candidate Profiling**: Generates a professional summary, experience level assessment, education breakdown, work experience summary, and key projects summary.
- **Deterministic Job Role Matching**: Evaluates candidate skills against 9+ predefined industry job roles (`job_roles.json`) using a transparent scoring formula:
  $$\text{Match Score} = \frac{\text{Matched Required Skills}}{\text{Total Required Skills}} \times 100$$
- **Skill Gap & Missing Skills Analysis**: Differentiates missing required skills from missing preferred skills, accompanied by tailored learning recommendations.
- **Interactive Dashboard**: Modern Streamlit UI with metric cards, color-coded skill chips, job compatibility tables, and instant page loads.
- **Privacy & In-Memory Processing**: Uploaded documents are processed entirely in-memory and are never stored on disk or in external databases.

---

## 🛠️ Technology Stack

- **Frontend & UI**: Streamlit, HTML5, Custom CSS
- **AI / LLM Integration**: Gemini API (`google-genai` SDK), OpenRouter API (`openai` SDK)
- **Text Extraction**: `pdfplumber` (PDF), `python-docx` (DOCX)
- **Environment & Utilities**: `python-dotenv`, `pandas`
- **Language**: Python 3.9+

---

## 📁 Project Structure

```text
ai-resume-screening/
│
├── app.py                      # Main Streamlit application entry point
├── requirements.txt            # Project Python dependencies
├── README.md                   # Complete project documentation
├── .env.example                # Template for environment variables
├── .gitignore                  # Excluded git tracking files
│
├── services/                   # Modular backend services
│   ├── __init__.py
│   ├── ai_provider.py          # Unified provider dispatcher & fallback layer
│   ├── openrouter_service.py   # OpenRouter integration with OpenAI SDK & retries
│   ├── gemini_service.py       # Google GenAI API integration
│   ├── resume_parser.py        # PDF & DOCX text extraction
│   ├── skill_matcher.py        # Deterministic scoring & implicit skill inference logic
│   └── validation.py           # File, text, and API key validation
│
├── data/
│   └── job_roles.json          # Predefined job roles and skill benchmarks
│
└── assets/
    └── styles.css              # Custom dashboard UI stylesheet
```

---

## ⚡ Quick Start (Local Setup)

### 1. Clone Repository & Navigate
```bash
git clone <repository-url>
cd ai-resume-screening
```

### 2. Create Virtual Environment
```bash
python -m venv venv
```

Activate virtual environment:
- **Windows**: `venv\Scripts\activate`
- **macOS / Linux**: `source venv/bin/activate`

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys
Create a `.env` file in the root directory by copying `.env.example`:
```bash
cp .env.example .env
```

Set your configuration and API keys:
```env
AI_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=openrouter/free
```

### 5. Launch Application
```bash
streamlit run app.py
```

---

## 🌐 How to Deploy (Get a Live Shareable Link)

Deploying to **Streamlit Community Cloud** is 100% free and takes less than 2 minutes:

### Step 1: Push Code to GitHub
```bash
git init
git add .
git commit -m "Initial commit for AI Resume Screening System"
git branch -M main
git remote add origin https://github.com/your-username/ai-resume-screening.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Community Cloud
1. Go to **[share.streamlit.io](https://share.streamlit.io/)** and sign in with your GitHub account.
2. Click **"New app"**.
3. Select your repository (`ai-resume-screening`), branch (`main`), and set Main file path to `app.py`.
4. Click **"Advanced settings..."** $\rightarrow$ **Secrets** and paste your API keys:
   ```toml
   AI_PROVIDER = "gemini"
   GEMINI_API_KEY = "your_actual_gemini_api_key"
   GEMINI_MODEL = "gemini-2.0-flash"
   OPENROUTER_API_KEY = "your_actual_openrouter_api_key"
   OPENROUTER_MODEL = "openrouter/free"
   ```
5. Click **"Deploy!"**. 

Once deployed, you will get a live public link like: `https://ai-resume-screening.streamlit.app` that you can share with anyone!
