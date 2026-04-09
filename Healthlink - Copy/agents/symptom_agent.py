"""
Symptom extraction agent.
Analyzes user input to extract symptoms, severity, and urgency.
Supports multi-turn conversations with clarifying questions.
"""
import logging
from typing import Optional, List, Dict, Any
import json
import re
from pathlib import Path

from core.llm import llm_generate, LLMClient
from core.rag import retrieve_relevant_docs, format_retrieval_context
from core.schemas import SymptomExtraction, Symptom
from config.settings import Settings


logger = logging.getLogger("healthlink.agents.symptom")
_SYMPTOM_KB_CACHE: Optional[list[dict[str, Any]]] = None


def _load_symptom_kb() -> list[dict[str, Any]]:
    global _SYMPTOM_KB_CACHE
    if _SYMPTOM_KB_CACHE is not None:
        return _SYMPTOM_KB_CACHE

    kb_path = Path(__file__).resolve().parents[1] / "data" / "symptoms_kb.json"
    if not kb_path.exists():
        _SYMPTOM_KB_CACHE = []
        return _SYMPTOM_KB_CACHE

    with kb_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    _SYMPTOM_KB_CACHE = data if isinstance(data, list) else []
    return _SYMPTOM_KB_CACHE


def _extract_duration(text: str) -> Optional[str]:
    patterns = [
        r"\b\d+\s*(day|days|week|weeks|month|months|hour|hours)\b",
        r"\b(since\s+\w+)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return None


def _severity_from_text(text: str) -> str:
    if any(word in text for word in ["severe", "intense", "worst"]):
        return "severe"
    if any(word in text for word in ["mild", "slight", "light"]):
        return "mild"
    return "moderate"


def _urgency_from_text_and_kb(text: str, matched_rows: list[dict[str, Any]]) -> str:
    emergency_words = ["chest pain", "difficulty breathing", "shortness of breath", "fainting", "seizure"]
    if any(word in text for word in emergency_words):
        return "emergency"

    for row in matched_rows:
        urgency = str(row.get("urgency", "")).lower()
        if "emergency" in urgency:
            return "emergency"
        if "urgent" in urgency:
            return "high"
        if "routine" in urgency:
            return "low"

    if "severe" in text:
        return "high"
    return "medium"


def _symptom_agent_offline(user_input: str) -> SymptomExtraction:
    text = user_input.lower()
    kb_rows = _load_symptom_kb()
    matched_rows: list[dict[str, Any]] = []
    symptoms: list[Symptom] = []
    duration = _extract_duration(text)
    severity = _severity_from_text(text)

    for row in kb_rows:
        symptom_name = str(row.get("symptom", "")).strip()
        if not symptom_name:
            continue
        if symptom_name.lower() in text:
            matched_rows.append(row)
            symptoms.append(
                Symptom(
                    name=symptom_name,
                    severity=severity,
                    duration=duration,
                )
            )

    if not symptoms:
        symptoms = [
            Symptom(
                name="general discomfort",
                severity=severity,
                duration=duration,
            )
        ]

    return SymptomExtraction(
        symptoms=symptoms,
        primary_complaint=user_input[:120],
        urgency_level=_urgency_from_text_and_kb(text, matched_rows),
        additional_context="Generated in offline mode using rule-based extraction.",
    )


def symptom_agent(
    user_input: str,
    llm_client: Optional[LLMClient] = None,
    settings: Optional[Settings] = None,
    use_rag: bool = True
) -> SymptomExtraction:
    """
    Extract symptoms and assess urgency from user input.

    This is a pure function that takes user input and returns structured symptom information.

    Args:
        user_input: User's description of their symptoms
        llm_client: LLM client instance (optional, will create if None)
        settings: Application settings (optional, will use global if None)
        use_rag: Whether to use RAG for additional context

    Returns:
        SymptomExtraction model with extracted symptoms and urgency assessment

    Example:
        >>> result = symptom_agent("I have a bad headache and fever for 3 days")
        >>> print(result.urgency_level)
        'medium'
    """
    logger.info("Symptom agent processing user input")

    if settings is None:
        from config.settings import get_settings
        settings = get_settings()

    if settings.offline_mode:
        logger.info("Symptom agent running in OFFLINE_MODE")
        return _symptom_agent_offline(user_input)

    context = ""
    if use_rag:
        try:
            retrieval_result = retrieve_relevant_docs(user_input, k=settings.rag_top_k, settings=settings)
            context = format_retrieval_context(retrieval_result, max_docs=3)
            logger.debug(f"Retrieved {len(retrieval_result.documents)} relevant documents")
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}. Continuing without context.")

    prompt = f"""Analyze the following patient complaint and extract structured symptom information.

Patient Input: "{user_input}"

Your task:
1. Identify all mentioned symptoms with their severity (mild, moderate, severe)
2. Note symptom duration if mentioned
3. Determine the primary health complaint
4. Assess urgency level based on symptoms:
   - emergency: Life-threatening symptoms (chest pain, difficulty breathing, severe bleeding, etc.)
   - high: Severe symptoms requiring prompt medical attention
   - medium: Moderate symptoms that should be evaluated soon
   - low: Mild symptoms for routine consultation

Be conservative with urgency assessment - if uncertain, err on the side of higher urgency.
"""

    try:
        result = llm_generate(
            prompt=prompt,
            schema=SymptomExtraction,
            temperature=0.2,
            context=context,
            client=llm_client
        )

        logger.info(
            f"Symptom extraction complete: "
            f"{len(result.symptoms)} symptoms, "
            f"urgency={result.urgency_level}"
        )

        return result

    except Exception as e:
        logger.error(f"Symptom agent failed: {e}", exc_info=True)
        return SymptomExtraction(
            symptoms=[],
            primary_complaint=user_input[:100],
            urgency_level="medium",
            additional_context="Error occurred during symptom analysis. Please consult a healthcare provider."
        )


async def symptom_agent_async(
    user_input: str,
    llm_client: Optional[LLMClient] = None,
    settings: Optional[Settings] = None,
    use_rag: bool = True
) -> SymptomExtraction:
    """
    Async version of symptom_agent.

    Note: Currently wraps synchronous implementation.
    For true async, LLM providers would need async clients.
    """
    return symptom_agent(user_input, llm_client, settings, use_rag)
