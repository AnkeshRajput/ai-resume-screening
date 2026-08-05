"""
OpenRouter AI service integration using the OpenAI Python SDK with exponential backoff retries.
"""

import json
import os
import time
from typing import Any, Dict
from openai import OpenAI

DEFAULT_OPENROUTER_MODEL = "openrouter/free"
MAX_RETRIES = 3
RETRY_DELAYS = [2, 4, 8]  # Exponential backoff in seconds


def build_analysis_prompt(resume_text: str) -> str:
    """
    Construct a structured prompt for OpenRouter LLM resume analysis.

    Args:
        resume_text (str): Cleaned resume plain text.

    Returns:
        str: Prompt text with JSON formatting rules.
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
7. CRITICAL: Do NOT invent or hallucinate skills, education, experience, or projects not present in the resume text.
8. If a section or detail is not present in the resume, return an empty list `[]` or `"Not specified"`.
9. You MUST return ONLY a raw valid JSON object.

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


def is_invalid_api_key_error(err_str: str) -> bool:
    """Check if error indicates an invalid API key or authentication failure."""
    lower_err = err_str.lower()
    return any(
        kw in lower_err
        for kw in [
            "401",
            "unauthorized",
            "invalid_api_key",
            "invalid api key",
            "authentication",
            "invalid_request_error",
        ]
    )


def analyze_resume_with_openrouter(
    resume_text: str,
    api_key: str = None,
    model_name: str = None,
) -> Dict[str, Any]:
    """
    Analyze candidate resume text via OpenRouter using the OpenAI Python SDK.
    Implements automatic exponential backoff retries (2s, 4s, 8s) for transient errors.

    Args:
        resume_text (str): Extracted resume plain text.
        api_key (str, optional): OpenRouter API key.
        model_name (str, optional): Target model name (default: openrouter/free).

    Returns:
        Dict[str, Any]: Sanitized dictionary matching application schema.

    Raises:
        ValueError: On missing key, authentication failure, or unrecoverable error.
    """
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key or not key.strip() or key == "your_openrouter_api_key_here":
        raise ValueError(
            "OpenRouter API key is missing. Please set OPENROUTER_API_KEY in your environment or .env file."
        )

    selected_model = (
        model_name
        or os.environ.get("OPENROUTER_MODEL")
        or DEFAULT_OPENROUTER_MODEL
    ).strip()

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=key,
        default_headers={
            "HTTP-Referer": "https://localhost:8501",
            "X-Title": "AI Resume Screening System",
        },
    )

    prompt = build_analysis_prompt(resume_text)
    messages = [
        {
            "role": "system",
            "content": "You are a precise resume parser. Output strictly valid JSON without explanation or markdown fences.",
        },
        {"role": "user", "content": prompt},
    ]

    last_exception = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=selected_model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.2,
            )

            if not response or not response.choices:
                raise ValueError("OpenRouter returned an empty choices payload.")

            content = response.choices[0].message.content
            if not content or not content.strip():
                raise ValueError("OpenRouter returned an empty message content.")

            # Sanitize content string if fences are returned
            cleaned_content = content.strip()
            if cleaned_content.startswith("```json"):
                cleaned_content = cleaned_content[7:]
            if cleaned_content.startswith("```"):
                cleaned_content = cleaned_content[3:]
            if cleaned_content.endswith("```"):
                cleaned_content = cleaned_content[:-3]

            cleaned_content = cleaned_content.strip()

            parsed_data = json.loads(cleaned_content)
            return sanitize_openrouter_output(parsed_data)

        except Exception as exc:
            last_exception = exc
            err_str = str(exc)

            # Do NOT retry invalid API key / authorization errors
            if is_invalid_api_key_error(err_str):
                raise ValueError(
                    "Invalid OpenRouter API key provided. Please verify your OPENROUTER_API_KEY credentials."
                )

            # If retries remain, execute exponential backoff pause
            if attempt < MAX_RETRIES:
                delay = RETRY_DELAYS[attempt]
                time.sleep(delay)
            else:
                break

    # If all attempts failed
    err_msg = str(last_exception)
    err_lower = err_msg.lower()

    if "429" in err_lower or "rate" in err_lower or "quota" in err_lower:
        raise ValueError(
            "OpenRouter API rate limit or quota exceeded. Please wait a moment or check your account."
        )
    elif "connect" in err_lower or "timeout" in err_lower:
        raise ValueError(
            "Network timeout connecting to OpenRouter. Please check your internet connection."
        )
    else:
        raise ValueError(f"OpenRouter analysis failed after retries: {err_msg}")


def sanitize_openrouter_output(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize and normalize dictionary structure returned from OpenRouter.

    Args:
        data (Dict[str, Any]): Raw JSON dictionary.

    Returns:
        Dict[str, Any]: Normalized dictionary.
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
        "role_recommendation_reason": "Based on extracted technical skills.",
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
            sanitized[field] = [str(item).strip() for item in val if item and str(item).strip()]
        else:
            sanitized[field] = []

    # Combine sub-technical fields into technical_skills set
    all_tech = set(sanitized["technical_skills"])
    for sub in ["programming_languages", "frameworks_and_libraries", "databases", "tools_and_platforms"]:
        for s in sanitized[sub]:
            all_tech.add(s)

    sanitized["technical_skills"] = sorted(list(all_tech))
    return sanitized
