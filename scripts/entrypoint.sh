#!/bin/bash

# FPL ML System Docker Entrypoint Script
set -e

echo "🚀 Starting FPL ML System..."

# Wait for database to be ready
echo "⏳ Waiting for database..."
while ! python -c "
import psycopg2
import os
try:
    conn = psycopg2.connect(os.getenv('DATABASE_URL'))
    conn.close()
    print('Database is ready!')
except:
    exit(1)
" 2>/dev/null; do
    echo "Database not ready, waiting..."
    sleep 2
done

# Wait for Redis to be ready
echo "⏳ Waiting for Redis..."
while ! python -c "
import redis
import os
try:
    r = redis.from_url(os.getenv('REDIS_URL', 'redis://redis:6379/0'))
    r.ping()
    print('Redis is ready!')
except:
    exit(1)
" 2>/dev/null; do
    echo "Redis not ready, waiting..."
    sleep 2
done

# Initialize database if needed
echo "🗄️ Initializing database..."
python -c "
from src.config.database import init_database
init_database()
print('Database initialized!')
" || echo "Database initialization skipped or failed"

# Run database migrations if any
echo "📊 Running migrations..."
python scripts/migrate.py || echo "No migrations to run"

# Initialize ML models if needed
echo "🤖 Checking ML models..."
python -c "
from src.models.ml_models import PlayerPredictor
try:
    predictor = PlayerPredictor()
    if not predictor.is_trained:
        print('ML models not trained - will train on first use')
    else:
        print('ML models ready!')
except Exception as e:
    print(f'ML models check failed: {e}')
"

# Setup logging
echo "📝 Setting up logging..."
mkdir -p logs
python -c "
from src.utils.logging import setup_logging
import os
setup_logging(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    structured=os.getenv('LOG_STRUCTURED', 'true').lower() == 'true'
)
print('Logging configured!')
"

# Start monitoring in background if enabled
if [ "${ENABLE_MONITORING:-true}" = "true" ]; then
    echo "📊 Starting system monitoring..."
    python -c "
from src.utils.monitoring import setup_monitoring
setup_monitoring()
print('Monitoring started!')
    " &
fi

# Cache warming (optional)
if [ "${WARM_CACHE:-false}" = "true" ]; then
    echo "♨️ Warming cache..."
    python -c "
from src.utils.cache import CacheWarmer
import asyncio
warmer = CacheWarmer()
asyncio.run(warmer.warm_bootstrap_data())
print('Cache warmed!')
    " || echo "Cache warming failed"
fi

# Print system info
echo "ℹ️ System Information:"
echo "   - Python: $(python --version)"
echo "   - Environment: ${ENVIRONMENT:-development}"
echo "   - Log Level: ${LOG_LEVEL:-INFO}"
echo "   - Database: ${DATABASE_URL:-Not configured}"
echo "   - Redis: ${REDIS_URL:-Not configured}"
echo "   - OpenAI API: ${OPENAI_API_KEY:+Configured}"

echo "✅ FPL ML System initialization complete!"

# Execute the main command
exec "$@"