FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
		POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

# Install system deps and build tools minimally
RUN apt-get update && apt-get install -y --no-install-recommends \
		build-essential curl ca-certificates && rm -rf /var/lib/apt/lists/*

# Copy and install Python deps first for better caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Copy application code
COPY . /app
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Healthcheck for container runtime
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
	CMD curl -f http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
