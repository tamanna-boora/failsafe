"""Rule-based intervention recommender, keyed on SHAP top features."""

from __future__ import annotations
from typing import Callable, Union
import pandas as pd


# Each rule: (feature_name, condition_fn, recommendation_text)
INTERVENTION_RULES: list[tuple[str, Callable[[float], bool], str]] = [
    (
        "absences",
        lambda v: v > 10,
        "High absenteeism detected — schedule an immediate attendance check-in "
        "with the academic advisor and explore the root cause (health, transport, motivation).",
    ),
    (
        "absences",
        lambda v: 5 < v <= 10,
        "Moderate absences noted — enrol the student in an attendance monitoring "
        "programme and set up bi-weekly check-ins.",
    ),
    (
        "studytime",
        lambda v: v < 2,
        "Very low weekly study time — refer the student to the 'Effective Study "
        "Skills' workshop and help them build a structured study timetable.",
    ),
    (
        "failures",
        lambda v: v > 0,
        f"Previous course failures on record — connect the student with a subject "
        "tutor and consider remedial/bridging classes before the next assessment.",
    ),
    (
        "famsup",
        # LabelEncoder sorts alphabetically, so no=0, yes=1
        lambda v: v == 0,
        "No family educational support — arrange a parent/guardian meeting to "
        "discuss the student's progress and strategies to support study at home.",
    ),
    (
        "Medu",
        lambda v: v < 2,
        "Mother has limited formal education — connect the student with a "
        "structured mentorship programme to supplement home academic support.",
    ),
    (
        "Fedu",
        lambda v: v < 2,
        "Father has limited formal education — connect the student with a "
        "structured mentorship programme to supplement home academic support.",
    ),
    (
        "Dalc",
        lambda v: v >= 3,
        "High weekday alcohol consumption reported — refer to student wellness "
        "services and arrange a confidential counselling session.",
    ),
    (
        "Walc",
        lambda v: v >= 4,
        "High weekend alcohol consumption reported — refer to student wellness "
        "services and arrange a confidential counselling session.",
    ),
    (
        "health",
        lambda v: v <= 2,
        "Poor self-reported health — refer the student to campus health services "
        "for a check-up and explore whether health issues are affecting attendance.",
    ),
    (
        "goout",
        lambda v: v >= 4,
        "High social going-out frequency — recommend a time-management coaching "
        "session to help balance social activities with academic responsibilities.",
    ),
    (
        "G1",
        lambda v: v < 8,
        "Very low first-period grade — trigger an immediate academic intervention: "
        "diagnostic test, targeted tutoring, and a learning support plan.",
    ),
    (
        "G2",
        lambda v: v < 8,
        "Very low second-period grade — escalate to learning support team for an "
        "intensive intervention before final assessments.",
    ),
    (
        "G1",
        lambda v: 8 <= v < 10,
        "Below-pass first-period grade — set up weekly progress meetings with a "
        "subject teacher to address knowledge gaps early.",
    ),
    (
        "G2",
        lambda v: 8 <= v < 10,
        "Below-pass second-period grade — set up weekly progress meetings with a "
        "subject teacher and review study strategies.",
    ),
    (
        "internet",
        # LabelEncoder sorts alphabetically, so no=0, yes=1
        lambda v: v == 0,
        "No home internet access — connect the student with campus digital "
        "resources (library WiFi, loaner devices) to support self-study.",
    ),
    (
        "schoolsup",
        # LabelEncoder sorts alphabetically, so no=0, yes=1
        lambda v: v == 0,
        "Not enrolled in extra school support — recommend signing up for "
        "after-school tutoring or peer study groups.",
    ),
    (
        "paid",
        # LabelEncoder sorts alphabetically, so no=0, yes=1
        lambda v: v == 0,
        "Not taking paid extra classes — if affordable, explore supplementary "
        "paid coaching in the student's weakest subjects.",
    ),
    (
        "freetime",
        lambda v: v >= 4,
        "High unstructured free time — encourage enrolment in structured "
        "extracurricular activities (clubs, sports) to build discipline.",
    ),
]

# used when rule matches don't reach the minimum of 3
_FALLBACKS = [
    "Schedule a general academic support session with the student's personal tutor.",
    "Provide the student with a curated list of online learning resources for their subjects.",
    "Pair the student with a higher-performing peer mentor for collaborative study.",
    "Encourage the student to attend all available office hours and revision sessions.",
    "Set SMART academic goals with the student and review progress fortnightly.",
]


def generate_interventions(
    student_row: Union[dict, "pd.Series"],
    top_features: list[str],
    max_interventions: int = 5,
) -> list[str]:
    """Return 3–5 personalised intervention strings for one at-risk student."""
    if isinstance(student_row, pd.Series):
        student_row = student_row.to_dict()

    seen: set[str] = set()   # de-duplicate by recommendation text
    results: list[str] = []

    # check SHAP top features first so the most impactful interventions lead
    priority_features = set(top_features[:10])
    for feat, condition, recommendation in INTERVENTION_RULES:
        if feat not in priority_features:
            continue
        val = student_row.get(feat)
        if val is None:
            continue
        try:
            if condition(val) and recommendation not in seen:
                seen.add(recommendation)
                results.append(recommendation)
        except Exception:
            continue
        if len(results) >= max_interventions:
            break

    if len(results) < 3:
        for feat, condition, recommendation in INTERVENTION_RULES:
            if feat in priority_features:
                continue
            val = student_row.get(feat)
            if val is None:
                continue
            try:
                if condition(val) and recommendation not in seen:
                    seen.add(recommendation)
                    results.append(recommendation)
            except Exception:
                continue
            if len(results) >= max_interventions:
                break

    for fallback in _FALLBACKS:
        if len(results) >= 3:
            break
        if fallback not in seen:
            seen.add(fallback)
            results.append(fallback)

    return results[:max_interventions]
