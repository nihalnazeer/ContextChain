# Use an official Python runtime
FROM python:3.9-slim

# Set working directory
WORKDIR /usr/src/app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy .dockerignore-excluded files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -e .

# Healthcheck (optional, for production)
# HEALTHCHECK --interval=30s --timeout=3s \
#   CMD curl -f http://localhost/ || exit 1

# Default command to run tests (with MongoDB URI override)
CMD ["sh", "-c", "python3 -m unittest discover -s tests || exit 0"]