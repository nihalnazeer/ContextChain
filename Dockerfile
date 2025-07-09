# Use an official Python runtime
FROM python:3.9-slim

# Set working directory
WORKDIR /usr/src/app

# Copy files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set entrypoint if needed (example if you have a CLI)
# ENTRYPOINT ["python", "-m", "app.cli"]

# For now, if running test or service:
CMD ["python3", "-m", "unittest", "discover", "-s", "tests"]
