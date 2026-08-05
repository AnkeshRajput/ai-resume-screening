"""
Gemini service module for AI-powered resume analysis using the official google-genai SDK.
"""

import json
import os
from typing import Any, Dict
from google import genai
from google.genai import types

# Default and fallback Gemini models for robust compatibility
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
FALLBACK_GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash-lite",
]


def build_analysis_prompt(resume_text: str) -> str:
    """
    Construct a precise, structured prompt for Gemini API to analyze the candidate profile.

    Args:
        resume_text (str): Extracted plain text from candidate's resume.

    Returns:
        str: Prompt instructions for Gemini LLM.
    """
    return f"""
You are an expert AI Resume Analyst and Senior Technical Recruiter.
Analyze the candidate's resume text below and extract comprehensive insights into a structured JSON format.

--- RESUME TEXT START ---
{resume_text}
--- RESUME TEXT END ---

INSTRUCTIONS:
1. Extract technical skills, soft skills, programming languages, frameworks & libraries, databases, tools & platforms.
2. Analyze education background, professional work experience, and key projects.
3. Determine the candidate's overall experience level (e.g., "Fresher", "Junior (1-2 yrs)", "Mid-Level (3-5 yrs)", "Senior (5+ yrs)").
4. Synthesize a concise, professional profile summary (2-4 sentences).
5. Recommend the single most suitable job role based on their skillset and experience.
6. Provide a clear, factual reason explaining why this job role is suitable.
7. CRITICAL: Do NOT invent or hallucinate information not present in the resume text.
8. If a section or detail is not present in the resume, return an empty list `[]` or `"Not specified"`.
9. You MUST return ONLY a raw valid JSON object matching the exact format specified below.

JSON OUTPUT STRUCTURE REQUIREMENT:
{{
  "candidate_name": "Full Name or Not specified",
  "experience_level": "Fresher / Junior / Mid-Level / Senior",
  "education_summary": "Concise summary of degrees, institutions, and graduation years",
  "profile_summary": "Concise professional summary summarizing candidate's background",
  "technical_skills": ["Python", "React", "Node.js"],
  "soft_skills": ["Problem-solving", "Teamwork", "Communication"],
  "programming_languages": ["Python", "JavaScript"],
  "frameworks_and_libraries": ["React", "Streamlit", "Express"],
  "databases": ["MongoDB", "PostgreSQL"],
  "tools_and_platforms": ["Git", "GitHub", "Docker"],
  "work_experience_summary": "Summary of employment history or 'No professional experience mentioned'",
  "project_summary": "Summary of notable projects, technologies used, and key accomplishments",
  "recommended_job_role": "Title of the most fitting job role (e.g., Full-Stack Developer)",
  "role_recommendation_reason": "Clear explanation for why this candidate fits this role"
}}
"""


def analyze_resume_with_gemini(
    resume_text: str,
    api_key: str = None,
    model_name: str = None,
) -> Dict[str, Any]:
    """
    Send resume text to Gemini API using the google-genai SDK and parse response.

    Args:
        resume_text (str): Cleaned resume text.
        api_key (str, optional): Gemini API key. Defaults to GEMINI_API_KEY env var.
        model_name (str, optional): Gemini model name. Defaults to GEMINI_MODEL env var or gemini-2.5-flash.

    Returns:
        Dict[str, Any]: Structured JSON dict of resume analysis.

    Raises:
        ValueError: For missing configuration, quota errors, network issues, or invalid JSON.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key or not key.strip() or key == "your_gemini_api_key_here":
        raise ValueError(
            "Gemini API key is missing. Please set GEMINI_API_KEY in your environment or .env file."
        )

    primary_model = (
        model_name
        or os.environ.get("GEMINI_MODEL")
        or DEFAULT_GEMINI_MODEL
    ).strip()

    # Build candidate models list avoiding duplicates while respecting user's primary choice
    models_to_try = [primary_model]
    for fb in FALLBACK_GEMINI_MODELS:
        if fb not in models_to_try:
            models_to_try.append(fb)

    client = genai.Client(api_key=key)
    prompt = build_analysis_prompt(resume_text)

    last_exception = None
    response = None

    for model_candidate in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_candidate,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                ),
            )
            if response and response.text:
                # Save working model in env if updated
                os.environ["GEMINI_MODEL"] = model_candidate
                break
        except Exception as exc:
            last_exception = exc
            err_msg_lower = str(exc).lower()
            if "404" in err_msg_lower or "not_found" in err_msg_lower or "no longer available" in err_msg_lower:
                # Model unavailable, continue to next fallback candidate
                continue
            else:
                # For non-404 errors (quota, key, network), raise immediately
                break

    if not response or not response.text:
        if last_exception:
            err_msg = str(last_exception)
            err_msg_lower = err_msg.lower()
            if "429" in err_msg_lower or "quota" in err_msg_lower or "rate" in err_msg_lower:
                raise ValueError(
                    "Gemini API rate limit or quota exceeded. Please wait a few moments or check your API quota."
                )
            elif "api_key" in err_msg_lower or ("invalid" in err_msg_lower and "key" in err_msg_lower):
                raise ValueError(
                    "Invalid Gemini API key provided. Please verify your API key credentials."
                )
            elif "connect" in err_msg_lower or "timeout" in err_msg_lower:
                raise ValueError(
                    "Network timeout connecting to Gemini API. Please check your network connection."
                )
            else:
                raise ValueError(f"Gemini API analysis failed: {err_msg}")
        else:
            raise ValueError("Received an empty response from Gemini API.")

    try:
        response_text = response.text.strip()

        # Sanitize code fences if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        response_text = response_text.strip()

        # Parse JSON
        parsed_data = json.loads(response_text)

        # Normalize and validate default schema structure
        normalized_data = sanitize_gemini_output(parsed_data)
        return normalized_data

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to parse Gemini response as JSON: {str(exc)}. Raw response: {response_text[:200]}"
        )


def sanitize_gemini_output(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure all expected keys exist with valid types and default fallbacks.

    Args:
        data (Dict[str, Any]): Raw decoded JSON dictionary.

    Returns:
        Dict[str, Any]: Sanitized dictionary matching target schema.
    """
    list_fields = [
        "technical_skills",
        "soft_skills",
        "programming_languages",
        "frameworks_and_libraries",
        "databases",
        "tools_and_platforms",
    ]

    string_fields = {
        "candidate_name": "Not specified",
        "experience_level": "Fresher",
        "education_summary": "Not specified",
        "profile_summary": "No professional profile summary available.",
        "work_experience_summary": "No professional work experience mentioned.",
        "project_summary": "No specific projects mentioned.",
        "recommended_job_role": "Software Engineer",
        "role_recommendation_reason": "Based on extracted general technical skills.",
    }

    sanitized = {}

    for field, default_val in string_fields.items():
        val = data.get(field)
        if val is None or not str(val).strip() or str(val).strip().lower() == "null":
            sanitized[field] = default_val
        else:
            sanitized[field] = str(val).strip()

    for field in list_fields:
        val = data.get(field)
        if isinstance(val, list):
            # Clean string list items
            cleaned_list = [str(item).strip() for item in val if item and str(item).strip()]
            sanitized[field] = cleaned_list
        else:
            sanitized[field] = []

    # Combine all sub-technical skill categories into main technical_skills list if missing
    all_tech = set(sanitized["technical_skills"])
    for sub_field in ["programming_languages", "frameworks_and_libraries", "databases", "tools_and_platforms"]:
        for skill in sanitized[sub_field]:
            all_tech.add(skill)
    
    sanitized["technical_skills"] = sorted(list(all_tech))

    return sanitized
