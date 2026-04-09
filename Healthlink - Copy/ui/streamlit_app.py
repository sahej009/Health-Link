"""
HealthLink Streamlit UI
User-friendly interface for health assessment system.
"""
import streamlit as st
import requests
import json
from datetime import datetime, timedelta
import plotly.graph_objects as go


# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"

# Page config
st.set_page_config(
    page_title="HealthLink - Smart Health Management",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .symptom-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .doctor-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
    }
    .urgency-high {
        color: #ff4b4b;
        font-weight: bold;
    }
    .urgency-medium {
        color: #ffa500;
        font-weight: bold;
    }
    .urgency-low {
        color: #4caf50;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health():
    """Check if API is available."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_urgency_color(urgency_level):
    """Get color class for urgency level."""
    urgency_map = {
        "emergency": "urgency-high",
        "high": "urgency-high",
        "medium": "urgency-medium",
        "low": "urgency-low"
    }
    return urgency_map.get(urgency_level.lower(), "urgency-medium")


def display_symptom_analysis(symptom_data):
    """Display symptom analysis results."""
    st.subheader("📋 Symptom Analysis")

    # Primary complaint
    st.markdown(f"**Primary Complaint:** {symptom_data['primary_complaint']}")

    # Urgency level
    urgency_class = get_urgency_color(symptom_data['urgency_level'])
    st.markdown(
        f"**Urgency Level:** <span class='{urgency_class}'>{symptom_data['urgency_level'].upper()}</span>",
        unsafe_allow_html=True
    )

    # Symptoms
    st.markdown("**Identified Symptoms:**")
    for symptom in symptom_data['symptoms']:
        st.markdown(
            f"<div class='symptom-box'>"
            f"• <strong>{symptom['name']}</strong> - Severity: {symptom['severity']}"
            f"{f' (Duration: {symptom['duration']})' if symptom.get('duration') else ''}"
            f"</div>",
            unsafe_allow_html=True
        )

    # Additional context
    if symptom_data.get('additional_context'):
        st.info(f"ℹ️ {symptom_data['additional_context']}")


def display_doctor_recommendations(doctor_data):
    """Display doctor recommendations."""
    st.subheader("👨‍⚕️ Recommended Doctors")

    st.markdown(f"**Specialty Rationale:** {doctor_data['specialty_rationale']}")
    st.markdown(f"**Match Confidence:** {doctor_data['match_score']:.0%}")

    if not doctor_data['recommended_doctors']:
        st.warning("No doctors available at this time.")
        return

    # Display doctors in columns
    for idx, doctor in enumerate(doctor_data['recommended_doctors']):
        with st.container():
            st.markdown(
                f"""
                <div class='doctor-card'>
                    <h4>Dr. {doctor['name']}</h4>
                    <p><strong>Specialty:</strong> {doctor['specialty']}</p>
                    <p><strong>Experience:</strong> {doctor['experience_years']} years</p>
                    <p><strong>Rating:</strong> ⭐ {doctor['rating']}/5.0</p>
                    <p><strong>Availability:</strong> {doctor['availability']}</p>
                    <p><strong>Location:</strong> {doctor['location']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )


def display_scheduling(scheduling_data):
    """Display scheduling recommendations."""
    st.subheader("📅 Appointment Scheduling")

    if scheduling_data.get('scheduling_notes'):
        st.info(scheduling_data['scheduling_notes'])

    # Recommended slot
    if scheduling_data.get('recommended_slot'):
        slot = scheduling_data['recommended_slot']
        st.success(
            f"**Recommended Appointment:** {slot['doctor_name']} on "
            f"{slot['date']} at {slot['time']} ({slot['duration_minutes']} minutes)"
        )

    # Available slots
    if scheduling_data.get('available_slots'):
        st.markdown("**Other Available Slots:**")

        # Group by doctor
        slots_by_doctor = {}
        for slot in scheduling_data['available_slots'][:10]:  # Show first 10
            doctor = slot['doctor_name']
            if doctor not in slots_by_doctor:
                slots_by_doctor[doctor] = []
            slots_by_doctor[doctor].append(slot)

        for doctor, slots in slots_by_doctor.items():
            with st.expander(f"📅 {doctor}"):
                for slot in slots[:5]:  # Show 5 slots per doctor
                    st.write(f"• {slot['date']} at {slot['time']}")


def display_health_summary(summary_data):
    """Display health summary."""
    st.subheader("📝 Health Summary")

    # Main summary
    st.markdown(f"**Assessment Summary:**")
    st.write(summary_data['summary'])

    # Key findings
    st.markdown("**Key Findings:**")
    for finding in summary_data['key_findings']:
        st.markdown(f"• {finding}")

    # Recommended actions
    st.markdown("**Recommended Actions:**")
    for action in summary_data['recommended_actions']:
        st.markdown(f"✓ {action}")

    # Urgency assessment
    st.markdown(f"**Overall Urgency:** {summary_data['urgency_assessment'].upper()}")

    # Disclaimer
    st.warning(f"⚠️ **Disclaimer:** {summary_data['disclaimer']}")


def main():
    """Main Streamlit application."""

    # Header
    st.markdown("<h1 class='main-header'>🏥 HealthLink</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Smart Health Management System</p>", unsafe_allow_html=True)

    # Check API health
    if not check_api_health():
        st.error("⚠️ Cannot connect to HealthLink API. Please ensure the backend server is running.")
        st.info("Start the backend server with: `python main.py`")
        return

    # Sidebar
    with st.sidebar:
        st.header("About HealthLink")
        st.write(
            "HealthLink uses AI to analyze your symptoms and recommend "
            "appropriate healthcare providers."
        )

        st.markdown("---")
        st.subheader("How it works:")
        st.write("1. Describe your symptoms")
        st.write("2. Get AI-powered analysis")
        st.write("3. Receive doctor recommendations")
        st.write("4. Schedule an appointment")

        st.markdown("---")
        st.caption("⚠️ This is not a substitute for professional medical advice.")

    # Main content
    st.markdown("### Tell us about your health concern")

    # Input form
    with st.form("assessment_form"):
        user_input = st.text_area(
            "Describe your symptoms in detail:",
            placeholder="Example: I have had a severe headache for 3 days, along with fever and sensitivity to light...",
            height=150
        )

        col1, col2 = st.columns(2)

        with col1:
            user_id = st.text_input(
                "Your ID (optional):",
                placeholder="user123"
            )

        with col2:
            preferred_date = st.date_input(
                "Preferred appointment date (optional):",
                value=datetime.now() + timedelta(days=1)
            )

        submit_button = st.form_submit_button("Get Assessment", use_container_width=True)

    # Process submission
    if submit_button:
        if len(user_input.strip()) < 10:
            st.error("Please provide more details about your symptoms (at least 10 characters)")
            return

        # Show loading
        with st.spinner("Analyzing your symptoms... This may take a moment."):
            try:
                # Prepare request
                request_data = {
                    "user_input": user_input,
                    "user_id": user_id if user_id else None,
                    "preferred_date": preferred_date.strftime("%Y-%m-%d") if preferred_date else None
                }

                # Call API
                response = requests.post(
                    f"{API_BASE_URL}/assess",
                    json=request_data,
                    timeout=60
                )

                if response.status_code == 200:
                    result = response.json()

                    # Display results
                    st.success("✅ Assessment completed successfully!")

                    # Create tabs for organized display
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "📋 Symptoms",
                        "👨‍⚕️ Doctors",
                        "📅 Scheduling",
                        "📝 Summary"
                    ])

                    with tab1:
                        display_symptom_analysis(result['symptom_analysis'])

                    with tab2:
                        display_doctor_recommendations(result['doctor_recommendations'])

                    with tab3:
                        display_scheduling(result['scheduling_options'])

                    with tab4:
                        display_health_summary(result['health_summary'])

                    # Download results
                    st.markdown("---")
                    st.download_button(
                        label="📥 Download Full Assessment",
                        data=json.dumps(result, indent=2),
                        file_name=f"health_assessment_{result['request_id']}.json",
                        mime="application/json"
                    )

                else:
                    st.error(f"Error: {response.json().get('detail', 'Unknown error occurred')}")

            except requests.exceptions.Timeout:
                st.error("Request timed out. The system might be under heavy load. Please try again.")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to the server. Please ensure the API is running.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    main()
