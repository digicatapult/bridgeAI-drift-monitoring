# Use the official Python 3.12 image from the Docker Hub
FROM python:3.12-slim

# Install dependencies and Poetry
RUN apt-get update &&  \
    apt-get install -y --fix-missing build-essential git && \
    pip install --no-cache-dir poetry && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Configure Poetry to not use virtual environments
ENV POETRY_VIRTUALENVS_CREATE=false

# Copy the pyproject.toml and poetry.lock files
COPY pyproject.toml poetry.lock ./

# Install only non-development dependencies
RUN poetry install --without dev --no-root

# Copy the rest of the application code into the container
COPY src ./src

# Add source directory to python path
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Set the environment variable
ENV LOG_LEVEL=INFO

# Run the application
CMD ["python", "src/main.py"]
