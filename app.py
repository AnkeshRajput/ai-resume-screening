"""
AI Resume Screening & Job Recommendation System
Streamlit Frontend & Interactive Dashboard Application
"""

import os
import sys
import textwrap
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Ensure local services package can be imported
sys.path.insert(0, os.path.dirname(__file__))

from services.ai_provider import analyze_resume_with_ai, get_active_provider
from services.resume_parser import extract_resume_text
from services.skill_matcher import evaluate_job_match
from services.validation import (
    validate_ai_api_keys,
    validate_extracted_text,
    validate_uploaded_file,
)

# Load environment variables from .env if present
load_dotenv()

# Streamlit Page Setup
st.set_page_config(
    page_title="AI Resume Screening & Job Recommendation System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_css_content(css_file_path: str) -> str:
    """Read and cache CSS content."""
    if os.path.exists(css_file_path):
        with open(css_file_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


# Inject cached custom styles
CSS_PATH = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
css_content = load_css_content(CSS_PATH)
if css_content:
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


def render_sidebar():
    """Render application sidebar with AI provider status and API key management."""
    st.sidebar.image(
        "https://img.icons8.com/isometric/100/resume.png",
        width=65,
    )
    st.sidebar.title("⚙️ Configuration")

    # Active Provider Selector
    active_prov = get_active_provider()
    provider_choice = st.sidebar.selectbox(
        "AI Provider:",
        options=["gemini", "openrouter"],
        index=0 if active_prov == "gemini" else 1,
        help="Select AI provider. Gemini is recommended for fast 1-2 second responses.",
    )
    os.environ["AI_PROVIDER"] = provider_choice

    gemini_key_env = os.environ.get("GEMINI_API_KEY", "")
    has_gemini_key = bool(
        gemini_key_env and gemini_key_env != "your_gemini_api_key_here"
    )

    openrouter_key_env = os.environ.get("OPENROUTER_API_KEY", "")
    has_openrouter_key = bool(
        openrouter_key_env and openrouter_key_env != "your_openrouter_api_key_here"
    )

    st.sidebar.divider()
    st.sidebar.markdown("### 🔑 API Keys")

    # Gemini Key UI
    if has_gemini_key:
        st.sidebar.success("✓ Gemini API Key Active")
    else:
        st.sidebar.warning("⚠️ Gemini API Key Missing")
        user_gem_key = st.sidebar.text_input(
            "Enter Gemini API Key:",
            type="password",
            help="Get key from Google AI Studio (https://aistudio.google.com/)",
        )
        if user_gem_key:
            os.environ["GEMINI_API_KEY"] = user_gem_key
            st.sidebar.success("Gemini Key updated!")
            has_gemini_key = True

    # OpenRouter Key UI
    if has_openrouter_key:
        st.sidebar.success("✓ OpenRouter API Key Active")
    else:
        st.sidebar.warning("⚠️ OpenRouter Key Missing")
        user_or_key = st.sidebar.text_input(
            "Enter OpenRouter Key:",
            type="password",
            help="Get key from OpenRouter (https://openrouter.ai/keys)",
        )
        if user_or_key:
            os.environ["OPENROUTER_API_KEY"] = user_or_key
            st.sidebar.success("OpenRouter Key updated!")
            has_openrouter_key = True

    st.sidebar.divider()
    st.sidebar.subheader("🔒 Security Assurance")
    st.sidebar.info(
        "Resumes are processed strictly in-memory for analysis and are **never saved** to disk or database."
    )

    st.sidebar.caption("AI Resume Screening System v2.0")
    return has_gemini_key or has_openrouter_key


def render_header():
    """Render top hero header."""
    st.markdown(
        """
        <div class="main-header-container">
            <h1 class="main-header-title">AI Resume Screening & Job Recommendation System</h1>
            <p class="main-header-subtitle">
                Upload candidate resume (PDF/DOCX) for AI skill extraction, candidate profiling, job role matching, and skill gap analysis.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_skill_chips(skills: list, chip_class: str = "chip-tech") -> str:
    """Generate HTML string for skill chips without leading whitespace indentation."""
    if not skills:
        return "<p style='color: #64748B; font-style: italic;'>None specified</p>"
    chips_html = "".join(
        [f'<span class="skill-chip {chip_class}">{skill}</span>' for skill in skills]
    )
    return f'<div class="tag-container">{chips_html}</div>'


def safe_markdown(html_content: str):
    """Render HTML string safely without Streamlit multi-line code block indentation bugs."""
    dedented = textwrap.dedent(html_content).strip()
    st.markdown(dedented, unsafe_allow_html=True)


def main():
    keys_available = render_sidebar()
    render_header()

    # Upload Section
    st.markdown("### 📤 Upload Resume Document")
    col_upload, col_info = st.columns([2, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Choose a PDF or DOCX file",
            type=["pdf", "docx"],
            help="Supported formats: .pdf, .docx | Max size: 5 MB",
        )

    with col_info:
        safe_markdown(
            """
            <div class="custom-card">
                <div style="font-weight: 600; margin-bottom: 0.5rem; color: #1E293B;">Upload Guidelines:</div>
                <ul style="margin: 0; padding-left: 1.2rem; font-size: 0.88rem; color: #475569;">
                    <li>Formats: <b>.pdf</b>, <b>.docx</b></li>
                    <li>Max Size: <b>5 MB</b></li>
                    <li>Text-selectable documents only</li>
                </ul>
            </div>
            """
        )

    if uploaded_file is not None:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.success(f"✅ Ready: **{uploaded_file.name}** ({file_size_mb:.2f} MB)")

    # Analyze Button
    analyze_btn = st.button(
        "🚀 Analyze Resume",
        type="primary",
        use_container_width=True,
        disabled=(uploaded_file is None),
    )

    if analyze_btn:
        if uploaded_file is None:
            st.error("Please upload a resume file first.")
            return

        is_key_ok, key_msg = validate_ai_api_keys()
        if not is_key_ok:
            st.error(key_msg)
            return

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Step 1: File Validation
            status_text.markdown("⏳ **Step 1/5:** Validating uploaded file...")
            progress_bar.progress(20)
            is_valid_file, file_err = validate_uploaded_file(uploaded_file)
            if not is_valid_file:
                st.error(f"Validation Error: {file_err}")
                return

            # Step 2: Text Extraction
            status_text.markdown("⏳ **Step 2/5:** Extracting resume text...")
            progress_bar.progress(40)
            try:
                extracted_text = extract_resume_text(uploaded_file)
            except ValueError as ext_err:
                st.error(f"Text Extraction Error: {str(ext_err)}")
                return

            is_valid_text, text_err = validate_extracted_text(extracted_text)
            if not is_valid_text:
                st.error(f"Content Error: {text_err}")
                return

            # Step 3: Fast AI Resume Analysis
            status_text.markdown("⏳ **Step 3/5:** Analyzing profile with AI...")
            progress_bar.progress(60)
            try:
                ai_analysis, used_provider = analyze_resume_with_ai(extracted_text)
            except ValueError as ai_err:
                st.error(f"AI Error: {str(ai_err)}")
                return

            # Step 4: Skill Matching Engine
            status_text.markdown("⏳ **Step 4/5:** Matching candidate skills...")
            progress_bar.progress(80)
            cand_tech_skills = ai_analysis.get("technical_skills", [])
            recommended_role_llm = ai_analysis.get("recommended_job_role", "")

            match_result = evaluate_job_match(cand_tech_skills, recommended_role_llm)

            # Step 5: Preparing Dashboard
            status_text.markdown("⏳ **Step 5/5:** Building dashboard...")
            progress_bar.progress(100)
            status_text.empty()
            progress_bar.empty()

            st.session_state["analysis_result"] = {
                "ai_analysis": ai_analysis,
                "match_result": match_result,
                "used_provider": used_provider,
                "file_name": uploaded_file.name,
            }
            st.toast(f"Resume analysis completed via {used_provider}!", icon="🎉")

        except Exception as general_err:
            status_text.empty()
            progress_bar.empty()
            st.error(f"An error occurred: {str(general_err)}")
            return

    # Render Results Dashboard
    if "analysis_result" in st.session_state:
        results = st.session_state["analysis_result"]
        ai_data = results["ai_analysis"]
        match_data = results["match_result"]
        used_prov = results.get("used_provider", "AI")

        st.divider()

        st.markdown(
            f"## 📊 Screening Results: `{ai_data.get('candidate_name', 'Candidate')}`"
        )
        st.caption(f"Source file: **{results['file_name']}** | Provider: **{used_prov}**")

        # Metric Overview Cards
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric(
                label="Recommended Job Role",
                value=match_data["recommended_role"],
            )

        with m_col2:
            score = match_data["match_score"]
            st.metric(
                label="Job Match Score",
                value=f"{score}%",
                delta="Strong Match" if score >= 70 else ("Moderate" if score >= 40 else "Early"),
            )

        with m_col3:
            st.metric(
                label="Experience Level",
                value=ai_data.get("experience_level", "Fresher"),
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Profile & Education Summary
        col_summary, col_edu = st.columns([2, 1])
        with col_summary:
            with st.container():
                st.subheader("📝 Professional Profile Summary")
                st.write(ai_data.get("profile_summary"))

        with col_edu:
            with st.container():
                st.subheader("🎓 Education Summary")
                st.write(ai_data.get("education_summary"))

        st.divider()

        # Technical Skills Chips Breakdown
        st.subheader("🛠️ Technical Skills Breakdown")
        t_col1, t_col2 = st.columns(2)

        with t_col1:
            st.markdown("##### 💻 Programming Languages")
            safe_markdown(render_skill_chips(ai_data.get("programming_languages", []), "chip-prog"))

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### 📦 Frameworks & Libraries")
            safe_markdown(render_skill_chips(ai_data.get("frameworks_and_libraries", []), "chip-framework"))

        with t_col2:
            st.markdown("##### 🗄️ Databases")
            safe_markdown(render_skill_chips(ai_data.get("databases", []), "chip-database"))

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### ⚙️ Tools & Platforms")
            safe_markdown(render_skill_chips(ai_data.get("tools_and_platforms", []), "chip-tool"))

        st.divider()

        # Soft Skills Section
        st.subheader("🤝 Soft Skills")
        safe_markdown(render_skill_chips(ai_data.get("soft_skills", []), "chip-soft"))

        st.divider()

        # Experience & Projects Breakdown
        exp_col, proj_col = st.columns(2)
        with exp_col:
            st.subheader("💼 Work Experience Summary")
            st.write(ai_data.get("work_experience_summary"))

        with proj_col:
            st.subheader("🚀 Key Projects Summary")
            st.write(ai_data.get("project_summary"))

        st.divider()

        # Recommended Role & Skill Analysis
        st.subheader("🎯 Job Role Suitability & Analysis")
        st.markdown(f"#### **{match_data['recommended_role']}**")
        st.caption(match_data["role_description"])

        st.info(f"**Recommendation Rationale:** {match_data['recommendation_reason']}")

        st.markdown("**Matched Candidate Skills:**")
        safe_markdown(render_skill_chips(match_data["matched_required_skills"], "chip-matched"))

        st.divider()

        # Missing Skills Analysis
        st.subheader("⚠️ Skill Gap & Missing Skills Analysis")

        st.markdown(f"**Missing Required Skills for {match_data['recommended_role']}:**")
        safe_markdown(render_skill_chips(match_data["missing_required_skills"], "chip-missing-req"))

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Missing Preferred / Optional Skills:**")
        safe_markdown(render_skill_chips(match_data["missing_preferred_skills"], "chip-missing-pref"))

        st.markdown("<br>", unsafe_allow_html=True)
        st.warning(f"💡 **Learning Recommendation:**\n\n{match_data['learning_recommendation']}")

        # Role Compatibility Table
        with st.expander("📈 View Compatibility Scores Across All Job Roles"):
            scores_df = pd.DataFrame(
                list(match_data["all_role_scores"].items()),
                columns=["Job Role", "Match Score (%)"],
            ).sort_values(by="Match Score (%)", ascending=False)
            st.dataframe(scores_df, use_container_width=True, hide_index=True)

        # Disclaimer
        safe_markdown(
            """
            <div class="disclaimer-banner">
                📌 <b>Disclaimer:</b> This analysis is generated using AI and skill-based matching algorithms. 
                It is intended as career guidance and decision support, not as an automated hiring decision.
            </div>
            """
        )


if __name__ == "__main__":
    main()
