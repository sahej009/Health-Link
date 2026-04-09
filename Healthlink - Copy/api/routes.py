"""
API routes for HealthLink.
FastAPI endpoints for health assessment and related operations.
"""
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from config.settings import Settings, get_settings
from config.logging import get_logger
from core.llm import LLMClient, get_llm_client
from core.database import get_db_session, get_all_doctors, DoctorModel
from core.schemas import (
    HealthAssessmentRequest,
    HealthAssessmentResponse,
    HealthCheckResponse,
    ErrorResponse,
    DoctorDB
)
from core.orchestrator import orchestrate_health_assessment, validate_assessment_request
from utils.validators import validate_user_input


logger = logging.getLogger("healthlink.api")

router = APIRouter()


@router.get("/health", response_model=HealthCheckResponse, tags=["System"])
def health_check(settings: Settings = Depends(get_settings)):
    """
    Health check endpoint.

    Returns system status and service availability.
    """
    logger.info("Health check requested")

    services_status = {
        "llm": "healthy" if settings.gemini_api_key else "unavailable",
        "database": "healthy",
        "rag": "healthy"
    }

    return HealthCheckResponse(
        status="healthy",
        version="1.0.0",
        services=services_status
    )


@router.post(
    "/assess",
    response_model=HealthAssessmentResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    tags=["Assessment"]
)
def assess_health(
    request: HealthAssessmentRequest
):
    """
    Main health assessment endpoint.

    Processes user's health concerns through the complete agent pipeline:
    1. Symptom extraction and analysis
    2. Doctor recommendations
    3. Scheduling suggestions
    4. Comprehensive summary

    Args:
        request: Health assessment request with user input

    Returns:
        Complete health assessment with recommendations

    Example:
        ```json
        {
            "user_input": "I have a severe headache and fever for 3 days",
            "user_id": "user123",
            "preferred_date": "2024-02-15"
        }
        ```
    """
    logger.info(f"Health assessment requested for user: {request.user_id or 'anonymous'}")

    # Validate request
    is_valid, validation_error = validate_assessment_request(request)
    if not is_valid:
        logger.warning(f"Invalid request: {validation_error}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=validation_error
        )

    is_valid, validation_error = validate_user_input(request.user_input)
    if not is_valid:
        logger.warning(f"Invalid user input: {validation_error}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=validation_error
        )

    settings = get_settings()
    logger.info(f"Processing request with input: {request.user_input[:100]}")

    try:
        from core.database import get_db_session
        db_session_gen = get_db_session(settings)
        db_session = next(db_session_gen)
        
        llm_client = get_llm_client(settings)

        response = orchestrate_health_assessment(
            request=request,
            db_session=db_session,
            llm_client=llm_client,
            settings=settings
        )

        logger.info(f"Assessment complete [request_id={response.request_id}]")
        return response

    except Exception as e:
        logger.error(f"Assessment failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred processing your request. Please try again."
        )


@router.get("/test_simple", tags=["Debug"])
async def test_simple():
    """Simple test endpoint."""
    return {"message": "test works"}


@router.get(
    "/doctors",
    tags=["Doctors"]
)
def list_doctors():
    """
    List available doctors.

    Returns:
        List of doctors
    """
    logger.info("Listing all doctors")

    try:
        from core.database import get_db_session
        settings = get_settings()
        db_session = next(get_db_session(settings))
        doctors = get_all_doctors(db_session)

        doctor_responses = [
            DoctorDB(
                id=d.id,
                name=d.name,
                specialty=d.specialty,
                experience_years=d.experience_years,
                rating=d.rating,
                availability=d.availability,
                location=d.location,
                email=d.email,
                phone=d.phone
            )
            for d in doctors
        ]

        logger.info(f"Returning {len(doctor_responses)} doctors")
        return doctor_responses

    except Exception as e:
        logger.error(f"Failed to list doctors: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve doctors"
        )


@router.get(
    "/doctors/{doctor_id}",
    response_model=DoctorDB,
    responses={404: {"model": ErrorResponse}},
    tags=["Doctors"]
)
def get_doctor(
    doctor_id: int
):
    """
    Get doctor by ID.

    Args:
        doctor_id: Doctor's unique identifier

    Returns:
        Doctor information
    """
    logger.info(f"Getting doctor with ID: {doctor_id}")

    try:
        from core.database import get_doctor_by_id, get_db_session
        settings = get_settings()
        db_session_gen = get_db_session(settings)
        db_session = next(db_session_gen)
        doctor = get_doctor_by_id(db_session, doctor_id)

        if not doctor:
            logger.warning(f"Doctor not found: {doctor_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Doctor with ID {doctor_id} not found"
            )

        return DoctorDB(
            id=doctor.id,
            name=doctor.name,
            specialty=doctor.specialty,
            experience_years=doctor.experience_years,
            rating=doctor.rating,
            availability=doctor.availability,
            location=doctor.location,
            email=doctor.email,
            phone=doctor.phone
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get doctor: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve doctor"
        )


@router.get(
    "/specialties",
    response_model=List[str],
    tags=["Doctors"]
)
def list_specialties():
    """
    List all available medical specialties.

    Returns:
        List of unique specialties
    """
    logger.info("Listing medical specialties")

    try:
        from core.database import get_db_session
        settings = get_settings()
        db_session_gen = get_db_session(settings)
        db_session = next(db_session_gen)
        doctors = get_all_doctors(db_session)
        specialties = sorted(list(set(d.specialty for d in doctors)))

        logger.info(f"Returning {len(specialties)} specialties")
        return specialties

    except Exception as e:
        logger.error(f"Failed to list specialties: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve specialties"
        )
