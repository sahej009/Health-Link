"""
Tests for HealthLink API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
import json

from main import app
from core.schemas import HealthAssessmentResponse


# Load mock outputs
with open('tests/mock_llm_outputs.json', 'r') as f:
    MOCK_OUTPUTS = json.load(f)


@pytest.fixture
def client():
    """Test client fixture."""
    return TestClient(app)


@pytest.fixture
def mock_orchestrator():
    """Mock orchestrator response."""
    with patch('api.routes.orchestrate_health_assessment') as mock:
        # Create mock response
        mock_response = {
            "request_id": "test-request-id",
            "timestamp": "2024-01-01T00:00:00",
            "symptom_analysis": MOCK_OUTPUTS['symptom_extraction'],
            "doctor_recommendations": MOCK_OUTPUTS['doctor_recommendation'],
            "scheduling_options": MOCK_OUTPUTS['scheduling_recommendation'],
            "health_summary": MOCK_OUTPUTS['health_summary'],
            "metadata": {}
        }

        mock.return_value = HealthAssessmentResponse(**mock_response)
        yield mock


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
        assert 'version' in data
        assert 'services' in data


class TestAssessEndpoint:
    """Tests for assessment endpoint."""

    def test_assess_valid_request(self, client, mock_orchestrator):
        """Test valid assessment request."""
        request_data = {
            "user_input": "I have a severe headache and fever for 3 days",
            "user_id": "test-user",
            "preferred_date": "2024-02-15"
        }

        response = client.post("/api/v1/assess", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert 'request_id' in data
        assert 'symptom_analysis' in data
        assert 'doctor_recommendations' in data
        assert 'scheduling_options' in data
        assert 'health_summary' in data

    def test_assess_invalid_short_input(self, client):
        """Test assessment with too short input."""
        request_data = {
            "user_input": "headache"
        }

        response = client.post("/api/v1/assess", json=request_data)

        assert response.status_code == 400

    def test_assess_invalid_date_format(self, client):
        """Test assessment with invalid date format."""
        request_data = {
            "user_input": "I have a severe headache and fever for 3 days",
            "preferred_date": "invalid-date"
        }

        response = client.post("/api/v1/assess", json=request_data)

        assert response.status_code == 400

    def test_assess_missing_input(self, client):
        """Test assessment without user input."""
        request_data = {}

        response = client.post("/api/v1/assess", json=request_data)

        assert response.status_code == 422  # Validation error


class TestDoctorsEndpoint:
    """Tests for doctors endpoints."""

    @patch('api.routes.get_all_doctors')
    def test_list_all_doctors(self, mock_get_doctors, client):
        """Test listing all doctors."""
        # Setup mock
        mock_doctor = Mock()
        mock_doctor.id = 1
        mock_doctor.name = "Test Doctor"
        mock_doctor.specialty = "General Practice"
        mock_doctor.experience_years = 10
        mock_doctor.rating = 4.5
        mock_doctor.availability = "Mon-Fri"
        mock_doctor.location = "Test Location"
        mock_doctor.email = "test@test.com"
        mock_doctor.phone = "555-0100"

        mock_get_doctors.return_value = [mock_doctor]

        # Make request
        response = client.get("/api/v1/doctors")

        assert response.status_code == 200
        data = response.json()
        assert len(data) > 0
        assert data[0]['name'] == "Test Doctor"

    @patch('api.routes.get_doctors_by_specialty')
    def test_list_doctors_by_specialty(self, mock_get_by_specialty, client):
        """Test listing doctors filtered by specialty."""
        # Setup mock
        mock_doctor = Mock()
        mock_doctor.id = 1
        mock_doctor.name = "Test Cardiologist"
        mock_doctor.specialty = "Cardiology"
        mock_doctor.experience_years = 15
        mock_doctor.rating = 4.8
        mock_doctor.availability = "Mon-Fri"
        mock_doctor.location = "Heart Center"
        mock_doctor.email = "cardio@test.com"
        mock_doctor.phone = "555-0101"

        mock_get_by_specialty.return_value = [mock_doctor]

        # Make request
        response = client.get("/api/v1/doctors?specialty=Cardiology")

        assert response.status_code == 200
        data = response.json()
        assert len(data) > 0
        assert data[0]['specialty'] == "Cardiology"

    @patch('api.routes.get_doctor_by_id')
    def test_get_doctor_by_id(self, mock_get_by_id, client):
        """Test getting doctor by ID."""
        # Setup mock
        mock_doctor = Mock()
        mock_doctor.id = 1
        mock_doctor.name = "Test Doctor"
        mock_doctor.specialty = "General Practice"
        mock_doctor.experience_years = 10
        mock_doctor.rating = 4.5
        mock_doctor.availability = "Mon-Fri"
        mock_doctor.location = "Test Location"
        mock_doctor.email = "test@test.com"
        mock_doctor.phone = "555-0100"

        mock_get_by_id.return_value = mock_doctor

        # Make request
        response = client.get("/api/v1/doctors/1")

        assert response.status_code == 200
        data = response.json()
        assert data['id'] == 1
        assert data['name'] == "Test Doctor"

    @patch('api.routes.get_doctor_by_id')
    def test_get_doctor_not_found(self, mock_get_by_id, client):
        """Test getting non-existent doctor."""
        mock_get_by_id.return_value = None

        response = client.get("/api/v1/doctors/999")

        assert response.status_code == 404


class TestSpecialtiesEndpoint:
    """Tests for specialties endpoint."""

    @patch('api.routes.get_all_doctors')
    def test_list_specialties(self, mock_get_doctors, client):
        """Test listing medical specialties."""
        # Setup mock
        mock_doctor1 = Mock()
        mock_doctor1.specialty = "Cardiology"

        mock_doctor2 = Mock()
        mock_doctor2.specialty = "Neurology"

        mock_doctor3 = Mock()
        mock_doctor3.specialty = "Cardiology"  # Duplicate

        mock_get_doctors.return_value = [mock_doctor1, mock_doctor2, mock_doctor3]

        # Make request
        response = client.get("/api/v1/specialties")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2  # Should have unique specialties
        assert "Cardiology" in data
        assert "Neurology" in data


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root(self, client):
        """Test root endpoint."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert 'name' in data
        assert 'version' in data
        assert data['name'] == "HealthLink API"
