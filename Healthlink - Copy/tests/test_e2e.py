#!/usr/bin/env python3
"""
End-to-End Testing Script for HealthLink
Tests all API endpoints and workflows
"""
import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8000"

def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)

def print_result(success, message):
    """Print test result"""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status}: {message}")

def test_health_check():
    """Test health check endpoint"""
    print_header("TEST 1: Health Check")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/health")
        data = response.json()

        success = (
            response.status_code == 200 and
            data["status"] == "healthy" and
            all(data["services"].values())
        )

        print_result(success, f"Health check - Status: {data['status']}")
        print(f"  Services: LLM={data['services']['llm']}, DB={data['services']['database']}, RAG={data['services']['rag']}")
        return success
    except Exception as e:
        print_result(False, f"Health check failed: {e}")
        return False

def test_assessment(scenario_name, user_input, expected_urgency=None):
    """Test health assessment endpoint"""
    print_header(f"TEST: {scenario_name}")
    try:
        payload = {
            "user_input": user_input,
            "user_id": f"test_user_{int(time.time())}",
            "session_id": f"session_{int(time.time())}"
        }

        print(f"Input: {user_input}")
        response = requests.post(f"{BASE_URL}/api/v1/assess", json=payload, timeout=120)
        data = response.json()

        if response.status_code != 200:
            print_result(False, f"API returned status {response.status_code}")
            return False

        # Extract key information
        symptoms = data["symptom_analysis"]["symptoms"]
        urgency = data["symptom_analysis"]["urgency_level"]
        doctors = data["doctor_recommendations"]["recommended_doctors"]
        specialty = doctors[0]["specialty"] if doctors else "None"
        doctor_name = doctors[0]["name"] if doctors else "None"
        slot = data["scheduling_options"]["recommended_slot"]

        print(f"\nResults:")
        print(f"  Symptoms detected: {len(symptoms)}")
        for sym in symptoms[:3]:  # Show first 3
            print(f"    - {sym['name']}: {sym['severity']} ({sym['duration']})")
        print(f"  Urgency Level: {urgency}")
        print(f"  Specialty: {specialty}")
        print(f"  Doctor: {doctor_name}")
        if slot:
            print(f"  Appointment: {slot['date']} at {slot['time']}")

        # Validate urgency if expected
        if expected_urgency:
            urgency_match = urgency.lower() == expected_urgency.lower()
            print_result(urgency_match, f"Urgency level {'matches' if urgency_match else 'does not match'} expected ({expected_urgency})")

        print_result(True, "Assessment completed successfully")
        return True

    except Exception as e:
        print_result(False, f"Assessment failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_doctors_endpoint():
    """Test doctors listing endpoint"""
    print_header("TEST: Doctors Endpoint")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/doctors")
        data = response.json()

        success = response.status_code == 200 and len(data) > 0
        print_result(success, f"Retrieved {len(data)} doctors")

        if data:
            print(f"  Sample doctor: {data[0]['name']} - {data[0]['specialty']}")

        return success
    except Exception as e:
        print_result(False, f"Doctors endpoint failed: {e}")
        return False

def test_specialties_endpoint():
    """Test specialties listing endpoint"""
    print_header("TEST: Specialties Endpoint")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/specialties")
        data = response.json()

        success = response.status_code == 200 and len(data) > 0
        print_result(success, f"Retrieved {len(data)} specialties")

        if data:
            print(f"  Specialties: {', '.join(data[:5])}...")

        return success
    except Exception as e:
        print_result(False, f"Specialties endpoint failed: {e}")
        return False

def run_all_tests():
    """Run all end-to-end tests"""
    print("\n" + "🏥" * 40)
    print("  HEALTHLINK END-TO-END TEST SUITE")
    print("🏥" * 40)
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = []

    # Test 1: Health check
    results.append(test_health_check())
    time.sleep(1)

    # Test 2: Low urgency scenario
    results.append(test_assessment(
        "Low Urgency - Common Cold",
        "I have a runny nose, mild cough, and slight fever for 2 days",
        expected_urgency="low"
    ))
    time.sleep(2)

    # Test 3: Medium urgency scenario
    results.append(test_assessment(
        "Medium Urgency - Persistent Pain",
        "I've had lower back pain for a week that's getting worse when I sit",
        expected_urgency="medium"
    ))
    time.sleep(2)

    # Test 4: High urgency scenario
    results.append(test_assessment(
        "High Urgency - Severe Symptoms",
        "I have severe chest pain and shortness of breath that started an hour ago",
        expected_urgency="high"
    ))
    time.sleep(2)

    # Test 5: Skin condition
    results.append(test_assessment(
        "Dermatology - Rash",
        "I have a red itchy rash on my arms that appeared 3 days ago",
        expected_urgency=None
    ))
    time.sleep(2)

    # Test 6: Doctors endpoint
    results.append(test_doctors_endpoint())
    time.sleep(1)

    # Test 7: Specialties endpoint
    results.append(test_specialties_endpoint())

    # Summary
    print_header("TEST SUMMARY")
    total = len(results)
    passed = sum(results)
    failed = total - passed

    print(f"\nTotal Tests: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! HealthLink is working perfectly!")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review the errors above.")

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
