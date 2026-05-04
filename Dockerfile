FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory
WORKDIR /app

# Copy dependency files first to leverage Docker cache
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen

# Copy the rest of the project files
COPY . .

# Ensure python output is unbuffered
ENV PYTHONUNBUFFERED=1

# Expose ports commonly used by langgraph dev / studio
EXPOSE 8123 2024

# Set the default command to run langgraph dev
# We bind to 0.0.0.0 so it's accessible from outside the container
CMD ["uv", "run", "langgraph", "dev", "--host", "0.0.0.0"]
