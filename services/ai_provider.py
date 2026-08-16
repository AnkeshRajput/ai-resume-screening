"""
AI Provider Abstraction Layer & Fallback Dispatcher.
Supports OpenRouter as primary provider with automatic Gemini fallback support.
"""

import os
from typing import Any, Dict, Tuple
from services.gemini_service import analyze_resume_with_gemini
from services.openrouter_service import analyze_resume_with_openrouter


def get_active_provider() -> str:
    """
    Get configured AI provider string ('openrouter' or 'gemini').

    Returns:
        str: Configured provider name.
    """
    return os.environ.get("AI_PROVIDER", "openrouter").strip().lower()


def analyze_resume_with_ai(
    resume_text: str, provider: str = None
) -> Tuple[Dict[str, Any], str]:
    """
    Unified entry point for AI resume analysis with automatic provider fallback.

    Provider order:
    1. Primary provider specified by `provider` argument or `AI_PROVIDER` env var.
    2. Fallback provider if primary fails or lacks API key.
    3. Error if all providers fail.

    Args:
        resume_text (str): Extracted resume plain text.
        provider (str, optional): Target provider override.

    Returns:
        Tuple[Dict[str, Any], str]: (analysis_result_dict, successful_provider_name)

    Raises:
        ValueError: If all AI providers fail or lack valid API key credentials.
    """
    selected_provider = (provider or get_active_provider()).lower()
    errors = []

    # Sequence of providers to attempt based on primary selection
    if selected_provider == "gemini":
        provider_sequence = ["gemini", "openrouter"]
    else:
        provider_sequence = ["openrouter", "gemini"]

    for prov in provider_sequence:
        try:
            if prov == "openrouter":
                openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
                if not openrouter_key or openrouter_key == "your_openrouter_api_key_here":
                    errors.append("OpenRouter: API key is missing or default placeholder.")
                    continue
                
                res = analyze_resume_with_openrouter(resume_text)
                return res, "OpenRouter"

            elif prov == "gemini":
                gemini_key = os.environ.get("GEMINI_API_KEY", "")
                if not gemini_key or gemini_key == "your_gemini_api_key_here":
                    errors.append("Gemini: API key is missing or default placeholder.")
                    continue

                res = analyze_resume_with_gemini(resume_text)
                return res, "Gemini AI"

        except Exception as exc:
            errors.append(f"{prov.capitalize()}: {str(exc)}")
            continue

    # If both providers failed
    combined_err_msg = " | ".join(errors)
    raise ValueError(
        f"AI Resume Analysis failed across all available providers ({', '.join(provider_sequence)}). "
        f"Details: {combined_err_msg}"
    )

