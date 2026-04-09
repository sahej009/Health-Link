#!/bin/bash

# HealthLink Setup Script
# This script helps set up the HealthLink project quickly

set -e

echo "🏥 HealthLink Setup Script"
echo "=========================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Found Python $python_version"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip -q
echo "✓ pip upgraded"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt -q
echo "✓ Dependencies installed"
echo ""

# Download spaCy model
echo "Downloading spaCy model for PII detection..."
python -m spacy download en_core_web_sm -q
echo "✓ spaCy model downloaded"
echo ""

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✓ .env file created"
    echo ""
    echo "⚠️  IMPORTANT: Please edit .env and add your API keys!"
    echo ""
else
    echo "✓ .env file already exists"
    echo ""
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p data logs
echo "✓ Directories created"
echo ""

echo "=========================="
echo "✅ Setup Complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your API keys (GEMINI_API_KEY or OPENAI_API_KEY)"
echo "2. Run the backend: python main.py"
echo "3. In another terminal, run the UI: streamlit run ui/streamlit_app.py"
echo ""
echo "Optional:"
echo "- Run tests: pytest"
echo "- Use Docker: docker-compose up --build"
echo ""
echo "For more information, see README.md"
echo ""
