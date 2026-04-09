"""
Pydantic schemas for HealthLink.
All data validation and structured outputs use these models.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime, date
from pydantic import BaseModel, Field, validator


# ==================== Agent Input/Output Models ====================

class SymptomInput(BaseModel):
    """Input for symptom extraction."""
    user_input: str = Field(..., description="User's symptom description")
    user_id: Optional[str] = Field(None, description="Optional user identifier")


class Symptom(BaseModel):
    """Individual symptom with severity."""
    name: str = Field(..., description="Symptom name")
    severity: str = Field(..., description="Severity level: mild, moderate, severe")
    duration: Optional[str] = Field(None, description="How long symptom has been present")


class SymptomExtraction(BaseModel):
    """Output from symptom agent."""
    symptoms: List[Symptom] = Field(..., description="Extracted symptoms")
    primary_complaint: str = Field(..., description="Main health concern")
    urgency_level: str = Field(..., description="Urgency: low, medium, high, emergency")
    additional_context: Optional[str] = Field(None, description="Any additional relevant context")


class Doctor(BaseModel):
    """Doctor information."""
    name: str = Field(..., description="Doctor's full name")
    specialty: str = Field(..., description="Medical specialty")
    experience_years: int = Field(..., description="Years of experience")
    rating: float = Field(..., ge=0, le=5, description="Rating out of 5")
    availability: str = Field(..., description="General availability")
    location: Optional[str] = Field(None, description="Clinic location")


class DoctorRecommendation(BaseModel):
    """Output from doctor recommendation agent."""
    recommended_doctors: List[Doctor] = Field(..., description="List of recommended doctors")
    specialty_rationale: str = Field(..., description="Why this specialty was chosen")
    match_score: float = Field(..., ge=0, le=1, description="Overall match confidence")


class TimeSlot(BaseModel):
    """Available appointment time slot."""
    doctor_name: str = Field(..., description="Doctor's name")
    date: str = Field(..., description="Appointment date (YYYY-MM-DD)")
    time: str = Field(..., description="Appointment time (HH:MM)")
    duration_minutes: int = Field(default=30, description="Appointment duration")
    slot_id: str = Field(..., description="Unique slot identifier")


class SchedulingRecommendation(BaseModel):
    """Output from scheduling agent."""
    available_slots: List[TimeSlot] = Field(..., description="Available appointment slots")
    recommended_slot: Optional[TimeSlot] = Field(None, description="Best recommended slot")
    scheduling_notes: Optional[str] = Field(None, description="Additional scheduling information")


class HealthSummary(BaseModel):
    """Final health summary output."""
    summary: str = Field(..., description="Comprehensive health summary")
    key_findings: List[str] = Field(..., description="Key medical findings")
    recommended_actions: List[str] = Field(..., description="Recommended next steps")
    urgency_assessment: str = Field(..., description="Overall urgency level")
    disclaimer: str = Field(
        default="This is not a medical diagnosis. Please consult with healthcare professionals for medical advice.",
        description="Medical disclaimer"
    )


# ==================== Orchestrator Models ====================

class HealthAssessmentRequest(BaseModel):
    """Request for full health assessment."""
    user_input: str = Field(..., min_length=10, description="User's health concern description")
    user_id: Optional[str] = Field(None, description="User identifier for tracking")
    preferred_date: Optional[str] = Field(None, description="Preferred appointment date")
    preferred_location: Optional[str] = Field(None, description="Preferred doctor location")


class HealthAssessmentResponse(BaseModel):
    """Complete health assessment response."""
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    symptom_analysis: SymptomExtraction = Field(..., description="Symptom extraction results")
    doctor_recommendations: DoctorRecommendation = Field(..., description="Recommended doctors")
    scheduling_options: SchedulingRecommendation = Field(..., description="Scheduling information")
    health_summary: HealthSummary = Field(..., description="Comprehensive health summary")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


# ==================== Database Models ====================

class DoctorDB(BaseModel):
    """Doctor database model."""
    id: int
    name: str
    specialty: str
    experience_years: int
    rating: float
    availability: str
    location: str
    email: Optional[str] = None
    phone: Optional[str] = None


class AppointmentDB(BaseModel):
    """Appointment database model."""
    id: int
    user_id: str
    doctor_id: int
    appointment_date: date
    appointment_time: str
    status: str  # scheduled, completed, cancelled
    created_at: datetime
    notes: Optional[str] = None


# ==================== Utility Models ====================

class HealthCheckResponse(BaseModel):
    """API health check response."""
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = "1.0.0"
    services: Dict[str, str] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ==================== RAG Models ====================

class Document(BaseModel):
    """Document for RAG system."""
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    embedding: Optional[List[float]] = Field(None, description="Document embedding vector")


class RetrievalResult(BaseModel):
    """RAG retrieval result."""
    documents: List[Document] = Field(..., description="Retrieved documents")
    scores: List[float] = Field(..., description="Relevance scores")
    query: str = Field(..., description="Original query")
