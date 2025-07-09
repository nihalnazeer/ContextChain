# Use an official Python runtime
FROM python:3.9-slim

# Set working directory
WORKDIR /usr/src/app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy .dockerignore-excluded files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -e .

# Set entrypoint for CLI (optional, uncomment if needed)
# ENTRYPOINT ["contextchain"]

# Default command to run tests
CMD ["python3", "-m", "unittest", "discover", "-s", "tests"]