# 🏥 HealthLink - Smart Health Management System

An AI-powered health management system that analyzes symptoms, recommends doctors, and schedules appointments using Google Gemini AI and Pinecone RAG technology.

## 📋 Overview

HealthLink is a production-ready educational project demonstrating modern GenAI application architecture with:

- **FastAPI** backend with function-based multi-agent architecture
- **Google Gemini 2.0** for intelligent symptom analysis and recommendations
- **Pinecone** vector database for RAG (Retrieval-Augmented Generation)
- **LangChain 1.x** for LLM orchestration
- **SQLite** database for doctors and appointments
- **Docker & Cloud deployment** ready (Google Cloud Run, Hugging Face Spaces)

## ⚠️ Disclaimer

**This is an educational project and NOT a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified health providers with questions about medical conditions.**

## 🏗️ Architecture

### Monolithic Function-Based Design

```
HealthLink/
├── agents/              # Pure function agents (NO classes)
│   ├── symptom_agent.py    # Symptom extraction & urgency assessment
│   ├── doctor_agent.py     # Doctor matching & recommendations
│   ├── scheduling_agent.py # Appointment slot generation
│   └── summary_agent.py    # Health summary generation
├── core/                # Core infrastructure
│   ├── llm.py          # LangChain + Gemini LLM client
│   ├── rag.py          # Pinecone vector store
│   ├── database.py     # SQLAlchemy models
│   ├── schemas.py      # Pydantic models
│   └── orchestrator.py # Agent orchestration
├── api/                # FastAPI routes
├── config/             # Settings & logging
├── data/               # Data files
│   ├── doctors.csv     # 100 doctors across 30 specialties
│   ├── symptoms_kb.json # 200+ medical conditions
│   └── healthlink.db   # SQLite database (auto-generated)
└── ui/                 # Streamlit interface (optional)
```

### Agent Pipeline

1. **Symptom Agent** → Extracts symptoms, severity, duration, and urgency level
2. **Doctor Agent** → Matches symptoms to specialties and recommends top doctors
3. **Scheduling Agent** → Generates available appointment slots based on urgency
4. **Summary Agent** → Creates comprehensive health summary with recommendations

## 🚀 Quick Start

### Prerequisites

- **Python 3.12+** (Required - LangChain 1.x needs Python 3.10+)
- **Gemini API Key** ([Get it here](https://aistudio.google.com/apikey))
- **Pinecone API Key** ([Sign up here](https://www.pinecone.io/))
- **Docker** (Optional, for containerized deployment)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/programteam-cn/GenAI-Live-Course-Project-7-Smart-Health-Management-System.git
cd GenAI-Live-Course-Project-7-Smart-Health-Management-System
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env and add your API keys:
# GEMINI_API_KEY=your_gemini_api_key_here
# PINECONE_API_KEY=your_pinecone_api_key_here
```

5. **Run the application**
```bash
# Start FastAPI backend
python main.py

# The API will be available at:
# - API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
# - Health Check: http://localhost:8000/api/v1/health
```

6. **(Optional) Run Streamlit UI**
```bash
# In another terminal
streamlit run ui/streamlit_app.py
# Access at: http://localhost:8501
```

### Testing the API

Run the end-to-end test script:
```bash
python test_e2e.py
```

This will test all API endpoints with various medical scenarios.

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Build and start services
docker-compose up --build

# Access the API at http://localhost:8000
# Access API docs at http://localhost:8000/docs
```

### Individual Docker Build

```bash
# Build image
docker build -t healthlink .

# Run container
docker run -p 8000:8000 \
  -e GEMINI_API_KEY=your_gemini_key \
  -e PINECONE_API_KEY=your_pinecone_key \
  healthlink
```

## ☁️ Cloud Deployments

### Google Cloud Run

#### Automatic Deployment (GitHub Actions)

1. **Set up GitHub Secrets:**
   - `GCP_PROJECT_ID` - Your Google Cloud project ID
   - `GCP_SA_KEY` - Service account JSON key
   - `GEMINI_API_KEY` - Your Gemini API key
   - `PINECONE_API_KEY` - Your Pinecone API key

2. **Store secrets in Google Secret Manager:**
```bash
echo -n "your-gemini-key" | gcloud secrets create GEMINI_API_KEY --data-file=-
echo -n "your-pinecone-key" | gcloud secrets create PINECONE_API_KEY --data-file=-
```

3. **Push to main branch** - CI/CD will automatically deploy

#### Manual Deployment

```bash
# Authenticate
gcloud config set project YOUR_PROJECT_ID

# Deploy
gcloud run deploy healthlink \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --set-secrets "GEMINI_API_KEY=GEMINI_API_KEY:latest,PINECONE_API_KEY=PINECONE_API_KEY:latest"
```

### Hugging Face Spaces

See [README_HUGGINGFACE.md](README_HUGGINGFACE.md) for detailed Hugging Face deployment instructions.

Quick steps:
1. Create a new Docker Space on Hugging Face
2. Copy all files to your space
3. Add secrets: `GEMINI_API_KEY`, `PINECONE_API_KEY`
4. Push and deploy

## 📊 API Documentation

### Health Assessment

**POST** `/api/v1/assess`

Request:
```json
{
  "user_input": "I have severe headache and fever for 3 days",
  "user_id": "user123",
  "session_id": "session456",
  "preferred_date": "2025-02-15"
}
```

Response:
```json
{
  "request_id": "uuid",
  "timestamp": "2025-01-01T00:00:00",
  "symptom_analysis": {
    "symptoms": [
      {
        "name": "headache",
        "severity": "severe",
        "duration": "3 days",
        "description": "..."
      }
    ],
    "urgency_level": "high",
    "confidence_score": 0.95
  },
  "doctor_recommendations": {
    "recommended_doctors": [
      {
        "name": "Dr. Sarah Johnson",
        "specialty": "Neurology",
        "rating": 4.8,
        "experience_years": 15
      }
    ]
  },
  "scheduling_options": {
    "recommended_slot": {
      "doctor_name": "Dr. Sarah Johnson",
      "date": "2025-01-15",
      "time": "09:00 AM"
    },
    "available_slots": [...]
  },
  "health_summary": {
    "summary": "...",
    "next_steps": [...]
  }
}
```

### List Doctors

**GET** `/api/v1/doctors?specialty=Cardiology&limit=10`

### List Specialties

**GET** `/api/v1/specialties`

### Health Check

**GET** `/api/v1/health`

Response:
```json
{
  "status": "healthy",
  "timestamp": "2025-01-01T00:00:00",
  "version": "1.0.0",
  "services": {
    "llm": true,
    "database": true,
    "rag": true
  }
}
```

## 🧪 Testing

### Run End-to-End Tests

```bash
# Start the server first
python main.py

# In another terminal, run tests
python test_e2e.py
```

This tests:
- Health check endpoint
- Low urgency scenarios (common cold)
- Medium urgency scenarios (persistent pain)
- High urgency scenarios (chest pain)
- Various medical specialties
- Doctor and specialty endpoints

### Unit Tests (if available)

```bash
pytest tests/ -v --cov=. --cov-report=html
```

## 🔧 Configuration

### Required Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Google Gemini API key | ✅ Yes |
| `PINECONE_API_KEY` | Pinecone API key | ✅ Yes |

### Optional Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_MODEL_NAME` | Gemini model name | gemini-2.0-flash |
| `LLM_TEMPERATURE` | LLM temperature | 0.2 |
| `LLM_MAX_TOKENS` | Max output tokens | 2048 |
| `EMBEDDING_MODEL_NAME` | Embedding model | models/gemini-embedding-001 |
| `PINECONE_ENVIRONMENT` | Pinecone environment | us-east-1 |
| `PINECONE_INDEX_NAME` | Pinecone index name | healthlink |
| `RAG_TOP_K` | RAG retrieval count | 5 |
| `CHUNK_SIZE` | Text chunk size | 500 |
| `CHUNK_OVERLAP` | Chunk overlap | 50 |
| `DATABASE_URL` | Database URL | sqlite:///./data/healthlink.db |
| `LOG_LEVEL` | Logging level | INFO |

See [.env.example](.env.example) for complete configuration.

## 📁 Project Structure

```
healthlink/
├── agents/                    # Function-based agents
│   ├── symptom_agent.py      # Symptom extraction (no classes)
│   ├── doctor_agent.py       # Doctor recommendations
│   ├── scheduling_agent.py   # Appointment scheduling
│   └── summary_agent.py      # Health summary
├── api/
│   └── routes.py             # FastAPI endpoints
├── config/
│   ├── settings.py           # Pydantic settings
│   └── logging.py            # Simple Python logging
├── core/
│   ├── llm.py               # LangChain + Gemini client
│   ├── rag.py               # Pinecone vector store
│   ├── database.py          # SQLAlchemy models
│   ├── schemas.py           # Pydantic schemas
│   └── orchestrator.py      # Agent orchestration
├── data/
│   ├── doctors.csv          # 100 doctors dataset
│   ├── symptoms_kb.json     # 200+ medical conditions
│   └── healthlink.db        # SQLite database (auto-created)
├── ui/
│   └── streamlit_app.py     # Streamlit interface
├── utils/
│   ├── helpers.py           # Helper functions
│   └── validators.py        # Input validation
├── .github/workflows/
│   └── deploy.yaml          # CI/CD for Google Cloud Run
├── main.py                   # FastAPI application
├── test_e2e.py              # End-to-end tests
├── expand_datasets.py       # Dataset expansion utility
├── Dockerfile               # Docker for Cloud Run
├── Dockerfile.huggingface   # Docker for Hugging Face
├── docker-compose.yml       # Local Docker setup
└── requirements.txt         # Python dependencies
```

## 🎯 Features

### ✅ Implemented

- ✅ Multi-agent symptom analysis with urgency assessment
- ✅ RAG-enhanced medical knowledge using Pinecone
- ✅ Doctor recommendation engine (100 doctors, 30 specialties)
- ✅ Smart appointment scheduling based on urgency
- ✅ Comprehensive health summaries
- ✅ Google Gemini 2.0 integration
- ✅ LangChain 1.x with structured outputs
- ✅ Simple Python logging (no external dependencies)
- ✅ Input validation and error handling
- ✅ Streamlit UI (optional)
- ✅ Docker support (Cloud Run + Hugging Face)
- ✅ CI/CD pipeline (GitHub Actions)
- ✅ End-to-end testing
- ✅ Expanded datasets (100 doctors, 200+ conditions)

## 📚 Technology Stack

### Core Technologies

- **Python 3.12** - Programming language
- **FastAPI** - Web framework
- **Pydantic v2** - Data validation
- **SQLAlchemy** - Database ORM
- **SQLite** - Database

### AI/ML Stack

- **Google Gemini 2.0** - Large Language Model
- **LangChain 1.x** - LLM orchestration framework
- **langchain-google-genai** - Gemini integration
- **Pinecone** - Vector database for RAG
- **Gemini Embeddings** - Text embeddings

### Deployment

- **Docker** - Containerization
- **Google Cloud Run** - Serverless deployment
- **Hugging Face Spaces** - Alternative deployment
- **GitHub Actions** - CI/CD

### UI (Optional)

- **Streamlit** - Web interface
- **Plotly** - Interactive charts