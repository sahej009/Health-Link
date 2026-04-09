"""
Tests for HealthLink agents.
"""
import pytest
import json
from unittest.mock import Mock, patch, MagicMock

from core.schemas import SymptomExtraction, DoctorRecommendation, SchedulingRecommendation, HealthSummary
from agents.symptom_agent import symptom_agent
from agents.doctor_agent import doctor_agent
from agents.scheduling_agent import scheduling_agent
from agents.summary_agent import summary_agent


# Load mock outputs
with open('tests/mock_llm_outputs.json', 'r') as f:
    MOCK_OUTPUTS = json.load(f)


@pytest.fixture
def mock_llm_client():
    """Mock LLM client."""
    client = Mock()
    return client


@pytest.fixture
def mock_db_session():
    """Mock database session."""
    session = MagicMock()
    return session


@pytest.fixture
def mock_settings():
    """Mock settings."""
    settings = Mock()
    settings.rag_top_k = 5
    settings.llm_temperature = 0.2
    return settings


class TestSymptomAgent:
    """Tests for symptom agent."""

    def test_symptom_agent_basic(self, mock_llm_client, mock_settings):
        """Test basic symptom extraction."""
        with patch('agents.symptom_agent.llm_generate') as mock_generate:
            # Setup mock
            mock_generate.return_value = SymptomExtraction(**MOCK_OUTPUTS['symptom_extraction'])

            # Run agent
            result = symptom_agent(
                user_input="I have a severe headache and fever for 3 days",
                llm_client=mock_llm_client,
                settings=mock_settings,
                use_rag=False
            )

            # Assertions
            assert isinstance(result, SymptomExtraction)
            assert len(result.symptoms) > 0
            assert result.urgency_level in ['low', 'medium', 'high', 'emergency']
            assert result.primary_complaint != ""

    def test_symptom_agent_with_rag(self, mock_llm_client, mock_settings):
        """Test symptom extraction with RAG."""
        with patch('agents.symptom_agent.llm_generate') as mock_generate, \
             patch('agents.symptom_agent.retrieve_relevant_docs') as mock_rag:

            # Setup mocks
            mock_generate.return_value = SymptomExtraction(**MOCK_OUTPUTS['symptom_extraction'])
            mock_rag.return_value = Mock(documents=[])

            # Run agent
            result = symptom_agent(
                user_input="I have chest pain",
                llm_client=mock_llm_client,
                settings=mock_settings,
                use_rag=True
            )

            # Assertions
            assert isinstance(result, SymptomExtraction)
            mock_rag.assert_called_once()

    def test_symptom_agent_error_handling(self, mock_llm_client, mock_settings):
        """Test symptom agent error handling."""
        with patch('agents.symptom_agent.llm_generate') as mock_generate:
            # Setup mock to raise exception
            mock_generate.side_effect = Exception("LLM error")

            # Run agent - should not raise exception
            result = symptom_agent(
                user_input="test input",
                llm_client=mock_llm_client,
                settings=mock_settings,
                use_rag=False
            )

            # Should return default response
            assert isinstance(result, SymptomExtraction)
            assert result.urgency_level == "medium"


class TestDoctorAgent:
    """Tests for doctor agent."""

    def test_doctor_agent_basic(self, mock_llm_client, mock_db_session, mock_settings):
        """Test doctor recommendation."""
        with patch('agents.doctor_agent.llm_generate') as mock_generate, \
             patch('agents.doctor_agent.get_doctors_by_specialty') as mock_get_doctors:

            # Setup mocks
            from pydantic import BaseModel, Field

            class MockSpecialtyRec(BaseModel):
                recommended_specialty: str
                specialty_rationale: str
                match_score: float

            mock_generate.return_value = MockSpecialtyRec(
                recommended_specialty="General Practice",
                specialty_rationale="Test rationale",
                match_score=0.85
            )

            mock_doctor = Mock()
            mock_doctor.name = "Test Doctor"
            mock_doctor.specialty = "General Practice"
            mock_doctor.experience_years = 10
            mock_doctor.rating = 4.5
            mock_doctor.availability = "Mon-Fri"
            mock_doctor.location = "Test Location"

            mock_get_doctors.return_value = [mock_doctor]

            # Create symptom analysis
            symptom_analysis = SymptomExtraction(**MOCK_OUTPUTS['symptom_extraction'])

            # Run agent
            result = doctor_agent(
                symptom_analysis=symptom_analysis,
                db_session=mock_db_session,
                llm_client=mock_llm_client,
                settings=mock_settings
            )

            # Assertions
            assert isinstance(result, DoctorRecommendation)
            assert len(result.recommended_doctors) > 0
            assert 0 <= result.match_score <= 1


class TestSchedulingAgent:
    """Tests for scheduling agent."""

    def test_scheduling_agent_basic(self, mock_llm_client, mock_settings):
        """Test scheduling recommendation."""
        with patch('agents.scheduling_agent.llm_generate') as mock_generate:
            # Setup mock
            from pydantic import BaseModel, Field

            class MockSlotSelection(BaseModel):
                recommended_slot_id: str
                scheduling_notes: str

            # Create doctor recommendation
            doctor_rec = DoctorRecommendation(**MOCK_OUTPUTS['doctor_recommendation'])

            # Mock will be called but we'll test fallback
            mock_generate.side_effect = Exception("Test error")

            # Run agent
            result = scheduling_agent(
                doctor_recommendation=doctor_rec,
                urgency_level="medium",
                llm_client=mock_llm_client,
                settings=mock_settings
            )

            # Assertions
            assert isinstance(result, SchedulingRecommendation)
            assert len(result.available_slots) > 0

    def test_scheduling_urgency_levels(self, mock_llm_client, mock_settings):
        """Test different urgency levels."""
        doctor_rec = DoctorRecommendation(**MOCK_OUTPUTS['doctor_recommendation'])

        urgency_levels = ['low', 'medium', 'high', 'emergency']

        for urgency in urgency_levels:
            with patch('agents.scheduling_agent.llm_generate'):
                result = scheduling_agent(
                    doctor_recommendation=doctor_rec,
                    urgency_level=urgency,
                    llm_client=mock_llm_client,
                    settings=mock_settings
                )

                assert isinstance(result, SchedulingRecommendation)


class TestSummaryAgent:
    """Tests for summary agent."""

    def test_summary_agent_basic(self, mock_llm_client, mock_settings):
        """Test summary generation."""
        with patch('agents.summary_agent.llm_generate') as mock_generate:
            # Setup mock
            mock_generate.return_value = HealthSummary(**MOCK_OUTPUTS['health_summary'])

            # Create input data
            symptom_analysis = SymptomExtraction(**MOCK_OUTPUTS['symptom_extraction'])
            doctor_rec = DoctorRecommendation(**MOCK_OUTPUTS['doctor_recommendation'])
            scheduling_rec = SchedulingRecommendation(**MOCK_OUTPUTS['scheduling_recommendation'])

            # Run agent
            result = summary_agent(
                symptom_analysis=symptom_analysis,
                doctor_recommendation=doctor_rec,
                scheduling_recommendation=scheduling_rec,
                llm_client=mock_llm_client,
                settings=mock_settings
            )

            # Assertions
            assert isinstance(result, HealthSummary)
            assert result.summary != ""
            assert len(result.key_findings) > 0
            assert len(result.recommended_actions) > 0
            assert result.disclaimer != ""

    def test_summary_agent_error_handling(self, mock_llm_client, mock_settings):
        """Test summary agent error handling."""
        with patch('agents.summary_agent.llm_generate') as mock_generate:
            # Setup mock to raise exception
            mock_generate.side_effect = Exception("LLM error")

            # Create input data
            symptom_analysis = SymptomExtraction(**MOCK_OUTPUTS['symptom_extraction'])
            doctor_rec = DoctorRecommendation(**MOCK_OUTPUTS['doctor_recommendation'])
            scheduling_rec = SchedulingRecommendation(**MOCK_OUTPUTS['scheduling_recommendation'])

            # Run agent - should not raise exception
            result = summary_agent(
                symptom_analysis=symptom_analysis,
                doctor_recommendation=doctor_rec,
                scheduling_recommendation=scheduling_rec,
                llm_client=mock_llm_client,
                settings=mock_settings
            )

            # Should return default response
            assert isinstance(result, HealthSummary)
            assert result.disclaimer != ""
