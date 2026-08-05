"""
Validation service for input files, extracted text, and environment configurations.
"""

import os
from typing import Tuple

MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB
ALLOWED_EXTENSIONS = {".pdf", ".docx"}


def validate_uploaded_file(uploaded_file) -> Tuple[bool, str]:
    """
    Validate the uploaded file for type, extension, and file size limit.

    Args:
        uploaded_file: Streamlit UploadedFile object or file-like object.

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if uploaded_file is None:
        return False, "No file uploaded. Please select a PDF or DOCX resume."

    file_name = uploaded_file.name
    file_ext = os.path.splitext(file_name)[1].lower()

    if file_ext not in ALLOWED_EXTENSIONS:
        return (
            False,
            f"Unsupported file format '{file_ext}'. Please upload a PDF (.pdf) or Word document (.docx).",
        )

    # Check file size
    if uploaded_file.size > MAX_FILE_SIZE_BYTES:
        size_mb = uploaded_file.size / (1024 * 1024)
        return (
            False,
            f"File size ({size_mb:.2f} MB) exceeds the maximum limit of 5 MB. Please upload a smaller file.",
        )

    return True, ""


def validate_extracted_text(text: str) -> Tuple[bool, str]:
    """
    Validate that meaningful text was successfully extracted from the resume file.

    Args:
        text (str): The raw extracted string content.

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if not text or not text.strip():
        return (
            False,
            "No readable text could be extracted from the uploaded resume. "
            "If this is a scanned or image-only PDF, please upload a text-selectable PDF or a Word document.",
        )

    cleaned_text = text.strip()
    if len(cleaned_text) < 50:
        return (
            False,
            "The extracted text is too short (fewer than 50 characters). "
            "Please ensure the file contains a complete resume with readable text.",
        )

    return True, ""


def validate_ai_api_keys() -> Tuple[bool, str]:
    """
    Verify if at least one valid AI API key (OpenRouter or Gemini) is available.

    Returns:
        Tuple[bool, str]: (is_valid, status_or_error_message)
    """
    provider = os.environ.get("AI_PROVIDER", "openrouter").strip().lower()

    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()

    has_openrouter = bool(
        openrouter_key and openrouter_key != "your_openrouter_api_key_here"
    )
    has_gemini = bool(gemini_key and gemini_key != "your_gemini_api_key_here")

    if not has_openrouter and not has_gemini:
        return (
            False,
            "No AI API keys configured. Please provide an OPENROUTER_API_KEY or GEMINI_API_KEY in your .env file or sidebar.",
        )

    if provider == "openrouter" and not has_openrouter and has_gemini:
        return (
            True,
            "OpenRouter API key missing, but Gemini API key detected. System will use Gemini fallback.",
        )

    if provider == "gemini" and not has_gemini and has_openrouter:
        return (
            True,
            "Gemini API key missing, but OpenRouter API key detected. System will use OpenRouter fallback.",
        )

    return True, ""
