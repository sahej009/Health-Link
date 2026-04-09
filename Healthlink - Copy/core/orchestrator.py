"""
Orchestrator for HealthLink agents.
Coordinates execution of all agents in the correct sequence.
"""
import logging
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from core.llm import LLMClient
from core.schemas import HealthAssessmentRequest, HealthAssessmentResponse
from agents.symptom_agent import symptom_agent
from agents.doctor_agent import doctor_agent
from agents.scheduling_agent import scheduling_agent
from agents.summary_agent import summary_agent
from config.settings import Settings


logger = logging.getLogger("healthlink.orchestrator")


def orchestrate_health_assessment(
    request: HealthAssessmentRequest,
    db_session: Session,
    llm_client: Optional[LLMClient] = None,
    settings: Optional[Settings] = None
) -> HealthAssessmentResponse:
    """
    Orchestrate the complete health assessment pipeline.

    This is the main coordination function that calls all agents in sequence:
    1. Symptom Agent - Extract and analyze symptoms
    2. Doctor Agent - Recommend appropriate doctors
    3. Scheduling Agent - Generate appointment slots
    4. Summary Agent - Create comprehensive summary

    Args:
        request: Health assessment request from user
        db_session: Database session for doctor queries
        llm_client: LLM client instance (optional)
        settings: Application settings (optional)

    Returns:
        HealthAssessmentResponse with complete assessment

    Example:
        >>> request = HealthAssessmentRequest(user_input="I have a headache")
        >>> response = orchestrate_health_assessment(request, db_session)
        >>> print(response.health_summary.summary)
    """
    # Generate unique request ID
    request_id = str(uuid.uuid4())

    logger.info(f"Starting health assessment orchestration [request_id={request_id}]")

    # Get settings if not provided
    if settings is None:
        from config.settings import get_settings
        settings = get_settings()

    try:
        # Step 1: Symptom Analysis
        logger.info(f"[{request_id}] Step 1/4: Analyzing symptoms")
        symptom_analysis = symptom_agent(
            user_input=request.user_input,
            llm_client=llm_client,
            settings=settings,
            use_rag=True
        )
        logger.info(
            f"[{request_id}] Symptom analysis complete: "
            f"urgency={symptom_analysis.urgency_level}, "
            f"symptoms={len(symptom_analysis.symptoms)}"
        )

        # Step 2: Doctor Recommendation
        logger.info(f"[{request_id}] Step 2/4: Recommending doctors")
        doctor_recommendation = doctor_agent(
            symptom_analysis=symptom_analysis,
            db_session=db_session,
            llm_client=llm_client,
            settings=settings,
            max_recommendations=3
        )
        logger.info(
            f"[{request_id}] Doctor recommendation complete: "
            f"doctors={len(doctor_recommendation.recommended_doctors)}"
        )

        # Step 3: Scheduling
        logger.info(f"[{request_id}] Step 3/4: Generating scheduling options")
        scheduling_recommendation = scheduling_agent(
            doctor_recommendation=doctor_recommendation,
            urgency_level=symptom_analysis.urgency_level,
            llm_client=llm_client,
            settings=settings,
            preferred_date=request.preferred_date
        )
        logger.info(
            f"[{request_id}] Scheduling complete: "
            f"slots={len(scheduling_recommendation.available_slots)}"
        )

        # Step 4: Summary Generation
        logger.info(f"[{request_id}] Step 4/4: Generating health summary")
        health_summary = summary_agent(
            symptom_analysis=symptom_analysis,
            doctor_recommendation=doctor_recommendation,
            scheduling_recommendation=scheduling_recommendation,
            llm_client=llm_client,
            settings=settings
        )
        logger.info(f"[{request_id}] Summary generation complete")

        # Build complete response
        response = HealthAssessmentResponse(
            request_id=request_id,
            timestamp=datetime.utcnow(),
            symptom_analysis=symptom_analysis,
            doctor_recommendations=doctor_recommendation,
            scheduling_options=scheduling_recommendation,
            health_summary=health_summary,
            metadata={
                "user_id": request.user_id,
                "preferred_location": request.preferred_location,
                "processing_time_ms": 0  # Could add timing here
            }
        )

        logger.info(f"[{request_id}] Health assessment orchestration complete")
        return response

    except Exception as e:
        logger.error(
            f"[{request_id}] Orchestration failed: {e}",
            exc_info=True
        )
        raise


async def orchestrate_health_assessment_async(
    request: HealthAssessmentRequest,
    db_session: Session,
    llm_client: Optional[LLMClient] = None,
    settings: Optional[Settings] = None
) -> HealthAssessmentResponse:
    """
    Async version of orchestrate_health_assessment.

    Note: Currently wraps synchronous implementation.
    For true async, all agent functions would need async implementations.
    """
    return orchestrate_health_assessment(request, db_session, llm_client, settings)


def validate_assessment_request(request: HealthAssessmentRequest) -> tuple[bool, str]:
    """
    Validate health assessment request.

    Args:
        request: The request to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check minimum input length
    if len(request.user_input.strip()) < 10:
        return False, "Health concern description too short. Please provide more details."

    # Check for prohibited content (basic check)
    prohibited_keywords = ["test", "demo", "fake"]
    lower_input = request.user_input.lower()

    if any(keyword in lower_input for keyword in prohibited_keywords):
        logger.warning(f"Suspicious input detected: {request.user_input[:50]}")

    # Validate preferred date format if provided
    if request.preferred_date:
        try:
            datetime.strptime(request.preferred_date, "%Y-%m-%d")
        except ValueError:
            return False, "Invalid preferred_date format. Use YYYY-MM-DD."

    return True, ""
