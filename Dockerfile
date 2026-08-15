# ============================================================
# Lumen-Bailian Dockerfile
# Multi-stage build for Bailian High-Code Application deployment
# ============================================================

FROM python:3.12-slim

LABEL app="lumen-bailian"
LABEL description="AI-Powered Personalized Learning Platform on Bailian"

# Environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app

WORKDIR $APP_HOME

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create log directory
RUN mkdir -p /home/admin/app_logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Run the application
# Use standalone mode by default; switch to bailian mode for AgentScope deployment
CMD ["python", "main.py", "--mode", "standalone"]
