# Dockerfile for SatOps Demo
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including uv
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy dependency files
COPY pyproject.toml .python-version ./

# Install Python dependencies with uv (much faster than pip)
RUN uv sync --frozen

# Install Ollama for local LLM
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Start the application
CMD ["uv", "run", "python", "app.py"]
