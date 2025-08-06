#!/bin/bash

# FPL ML System - User Setup Script
echo "🚀 Setting up FPL ML System for user..."

# Create .env file with user's credentials
cat > .env << 'EOF'
# FPL User Configuration
FPL_TEAM_ID=3647781
FPL_EMAIL=aof_comsci@hotmail.com
FPL_PASSWORD=$3Qh9!AB

# Database Configuration (SQLite for development)
DATABASE_URL=sqlite:///data/fpl.db

# Cache Configuration (In-memory for development)
REDIS_URL=

# AI Configuration (Optional - add if you have OpenAI API key)
OPENAI_API_KEY=

# Logging Configuration
LOG_LEVEL=INFO
LOG_STRUCTURED=false
LOG_DIR=logs

# Application Configuration
ENVIRONMENT=development
DEBUG=false

# ML Model Configuration
MODEL_PATH=models
RETRAIN_THRESHOLD_MSE=0.005
MIN_TRAINING_SAMPLES=100

# Optimization Configuration
OPTIMIZATION_TIMEOUT_SECONDS=10
MIN_TRANSFER_GAIN_THRESHOLD=1.0
RISK_TOLERANCE=balanced

# API Rate Limiting
FPL_API_RATE_LIMIT=60
FPL_API_RATE_WINDOW=60

# Dashboard Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
DASHBOARD_TITLE="My FPL ML System"

# Development Configuration
WARM_CACHE=false
ENABLE_DEBUG_TOOLBAR=false
MOCK_FPL_API=false
EOF

echo "✅ .env file created successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Make sure you're in virtual environment: source venv/bin/activate"
echo "2. Install in development mode: pip install -e ."
echo "3. Test the system: fpl status"