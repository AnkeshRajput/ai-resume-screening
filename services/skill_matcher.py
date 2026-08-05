"""
Skill matching service for deterministic job role analysis, match score calculation,
and missing skills identification.
"""

import json
import os
from typing import Any, Dict, List, Set, Tuple

# Path to job roles dataset
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "job_roles.json")

# Synonyms and skill normalization mapping
SKILL_NORMALIZATION_MAP = {
    "js": "JavaScript",
    "javascript": "JavaScript",
    "ts": "TypeScript",
    "typescript": "TypeScript",
    "py": "Python",
    "python3": "Python",
    "nodejs": "Node.js",
    "node.js": "Node.js",
    "node js": "Node.js",
    "node": "Node.js",
    "reactjs": "React",
    "react.js": "React",
    "react": "React",
    "nextjs": "Next.js",
    "next.js": "Next.js",
    "mongo": "MongoDB",
    "mongodb": "MongoDB",
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "genai": "Generative AI",
    "generative ai": "Generative AI",
    "llms": "LLM",
    "llm": "LLM",
    "ml": "Machine Learning",
    "machine learning": "Machine Learning",
    "dl": "Deep Learning",
    "deep learning": "Deep Learning",
    "k8s": "Kubernetes",
    "kubernetes": "Kubernetes",
    "aws cloud": "AWS",
    "aws": "AWS",
    "html5": "HTML",
    "html": "HTML",
    "css3": "CSS",
    "css": "CSS",
    "docker containers": "Docker",
    "docker": "Docker",
    "github": "Git",
    "git": "Git",
    "scikit-learn": "Scikit-Learn",
    "sklearn": "Scikit-Learn",
    "tensorflow": "TensorFlow",
    "pytorch": "PyTorch",
}


def load_job_roles() -> Dict[str, Any]:
    """
    Load job roles and their skill requirements from job_roles.json.

    Returns:
        Dict[str, Any]: Dictionary of job roles.
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Job roles data file not found at '{DATA_PATH}'.")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def normalize_skill(skill: str) -> str:
    """
    Normalize skill string for case-insensitive matching and synonym unification.

    Args:
        skill (str): Raw skill string.

    Returns:
        str: Normalized skill name.
    """
    if not skill:
        return ""
    clean = skill.strip().lower()
    if clean in SKILL_NORMALIZATION_MAP:
        return SKILL_NORMALIZATION_MAP[clean]
    
    # Capitalize title format for clean display
    return skill.strip().title()


def get_normalized_skill_set(skills: List[str]) -> Tuple[Set[str], Dict[str, str]]:
    """
    Convert a list of raw skills into a set of lowercased normalized tokens,
    while maintaining a lookup map back to clean display strings.
    Automatically infers implicit foundational skills (e.g., React implies HTML & CSS).

    Args:
        skills (List[str]): List of candidate skills.

    Returns:
        Tuple[Set[str], Dict[str, str]]: (normalized_keys_set, key_to_display_map)
    """
    norm_set = set()
    key_map = {}

    for skill in skills:
        if not skill or not skill.strip():
            continue
        cleaned = skill.strip()
        norm_val = normalize_skill(cleaned)
        norm_key = norm_val.lower()
        norm_set.add(norm_key)
        if norm_key not in key_map:
            key_map[norm_key] = norm_val

    # Implicit Skill Implication Rules
    # If candidate possesses high-level web frameworks, infer underlying foundational web tech
    frontend_frameworks = {"react", "next.js", "vue", "angular", "tailwind css", "bootstrap", "redux"}
    if any(fw in norm_set for fw in frontend_frameworks):
        for impl_skill in ["HTML", "CSS", "JavaScript"]:
            impl_key = impl_skill.lower()
            norm_set.add(impl_key)
            if impl_key not in key_map:
                key_map[impl_key] = impl_skill

    # Python framework implication
    python_frameworks = {"django", "flask", "fastapi", "streamlit"}
    if any(pf in norm_set for pf in python_frameworks):
        norm_set.add("python")
        if "python" not in key_map:
            key_map["python"] = "Python"

    # Node implication
    if "node.js" in norm_set or "express" in norm_set:
        norm_set.add("javascript")
        if "javascript" not in key_map:
            key_map["javascript"] = "JavaScript"

    return norm_set, key_map


def match_skills_against_role(
    candidate_skills: List[str], role_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare candidate skills against a single job role's required and preferred skills.

    Args:
        candidate_skills (List[str]): Extracted candidate technical skills.
        role_data (Dict[str, Any]): Role definition dict from job_roles.json.

    Returns:
        Dict[str, Any]: Calculation results for the specified role.
    """
    cand_keys, cand_key_map = get_normalized_skill_set(candidate_skills)

    req_skills = role_data.get("required_skills", [])
    pref_skills = role_data.get("preferred_skills", [])

    matched_required = []
    missing_required = []
    for req in req_skills:
        req_norm = normalize_skill(req)
        req_key = req_norm.lower()
        if req_key in cand_keys:
            matched_required.append(cand_key_map.get(req_key, req_norm))
        else:
            missing_required.append(req_norm)

    matched_preferred = []
    missing_preferred = []
    for pref in pref_skills:
        pref_norm = normalize_skill(pref)
        pref_key = pref_norm.lower()
        if pref_key in cand_keys:
            matched_preferred.append(cand_key_map.get(pref_key, pref_norm))
        else:
            missing_preferred.append(pref_norm)

    total_required = len(req_skills)
    matched_count = len(matched_required)

    match_score = (
        round((matched_count / total_required) * 100, 1) if total_required > 0 else 0.0
    )

    return {
        "match_score": match_score,
        "matched_required_skills": matched_required,
        "missing_required_skills": missing_required,
        "matched_preferred_skills": matched_preferred,
        "missing_preferred_skills": missing_preferred,
        "total_required_skills_count": total_required,
        "description": role_data.get("description", ""),
    }


def evaluate_job_match(
    candidate_skills: List[str], gemini_recommended_role: str = None
) -> Dict[str, Any]:
    """
    Evaluate candidate technical skills against all job roles and determine final recommendation.

    Args:
        candidate_skills (List[str]): Candidate's extracted technical skills.
        gemini_recommended_role (str, optional): Role recommended by Gemini LLM.

    Returns:
        Dict[str, Any]: Comprehensive match and missing skills analysis payload.
    """
    job_roles = load_job_roles()

    role_evaluations = {}
    best_deterministic_role = None
    highest_score = -1.0

    for role_name, role_data in job_roles.items():
        eval_result = match_skills_against_role(candidate_skills, role_data)
        role_evaluations[role_name] = eval_result

        if eval_result["match_score"] > highest_score:
            highest_score = eval_result["match_score"]
            best_deterministic_role = role_name

    # Determine final recommended role using a hybrid strategy
    final_role = best_deterministic_role or "Software Engineer"
    recommendation_strategy = ""

    if gemini_recommended_role:
        # Check if Gemini role matches any predefined role name (case-insensitive)
        matched_predefined_name = None
        for r_name in job_roles.keys():
            if r_name.lower() in gemini_recommended_role.lower() or gemini_recommended_role.lower() in r_name.lower():
                matched_predefined_name = r_name
                break

        if matched_predefined_name:
            gemini_role_eval = role_evaluations[matched_predefined_name]
            # If Gemini's recommended role has a reasonable match score (>= 40%) or highest, prioritize it
            if gemini_role_eval["match_score"] >= highest_score - 15.0 or gemini_role_eval["match_score"] >= 40.0:
                final_role = matched_predefined_name
                recommendation_strategy = (
                    f"Recommended based on Gemini AI resume synthesis and validated with a "
                    f"{gemini_role_eval['match_score']}% skill match score."
                )

    if not recommendation_strategy:
        best_eval = role_evaluations[final_role]
        recommendation_strategy = (
            f"Determined via skill matching algorithm: Candidate matches {len(best_eval['matched_required_skills'])} "
            f"out of {best_eval['total_required_skills_count']} core required skills ({best_eval['match_score']}%)."
        )

    final_eval = role_evaluations[final_role]

    # Generate Learning Recommendation message
    learning_recommendation = generate_learning_recommendation(
        final_role,
        final_eval["missing_required_skills"],
        final_eval["missing_preferred_skills"],
        final_eval["match_score"],
    )

    return {
        "recommended_role": final_role,
        "match_score": final_eval["match_score"],
        "role_description": final_eval["description"],
        "recommendation_reason": recommendation_strategy,
        "matched_required_skills": final_eval["matched_required_skills"],
        "missing_required_skills": final_eval["missing_required_skills"],
        "matched_preferred_skills": final_eval["matched_preferred_skills"],
        "missing_preferred_skills": final_eval["missing_preferred_skills"],
        "learning_recommendation": learning_recommendation,
        "all_role_scores": {
            r: res["match_score"] for r, res in role_evaluations.items()
        },
    }


def generate_learning_recommendation(
    role_name: str,
    missing_required: List[str],
    missing_preferred: List[str],
    match_score: float,
) -> str:
    """
    Generate an actionable learning advice text based on missing skills.

    Args:
        role_name (str): Target job role title.
        missing_required (List[str]): Missing required skills.
        missing_preferred (List[str]): Missing preferred skills.
        match_score (float): Calculated match percentage.

    Returns:
        str: Constructive career guidance message.
    """
    if match_score == 100.0 and not missing_preferred:
        return f"🎉 Exceptional match! The candidate meets all required and preferred skill criteria for the {role_name} role."

    parts = []

    if match_score >= 70.0:
        parts.append(f"The candidate has a strong skill foundation for the {role_name} role ({match_score}% match).")
    elif match_score >= 40.0:
        parts.append(f"The candidate has a moderate skill foundation for the {role_name} role ({match_score}% match).")
    else:
        parts.append(f"The candidate is early in their journey toward the {role_name} role ({match_score}% match).")

    key_focus = missing_required[:3] if missing_required else missing_preferred[:3]

    if key_focus:
        skills_str = ", ".join(key_focus)
        parts.append(
            f"To boost job readiness, prioritizing learning and hands-on projects in **{skills_str}** is strongly recommended."
        )

    return " ".join(parts)
