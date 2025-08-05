# FPL ML System Production Dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    curl \
    git \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r fpl && useradd -r -g fpl fpl

# Copy requirements first for better Docker layer caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -r requirements-dev.txt

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/models /app/cache && \
    chown -R fpl:fpl /app

# Copy application code
COPY --chown=fpl:fpl . .

# Set permissions for entrypoint script
RUN chmod +x scripts/entrypoint.sh

# Install the package in development mode
RUN pip install -e .

# Create data directories
RUN mkdir -p data/raw data/processed data/models logs

# Switch to non-root user
USER fpl

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8501/_stcore/health', timeout=5)" || exit 1

# Expose ports
EXPOSE 8501 8000

# Set entrypoint
ENTRYPOINT ["scripts/entrypoint.sh"]

# Default command
CMD ["streamlit", "run", "src/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false", "--theme.base=light"]