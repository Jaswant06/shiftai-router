# The ShiftAI API server in a container. Ollama keeps running on the host,
# where it can use the GPU; the container only routes requests to it.
#
# On the host first:  ollama pull <models> && shiftai setup   (writes ~/.shiftai/hardware.json)
# Build:  docker build -t shiftai .
# Run:    docker run -p 8800:8800 -v ~/.shiftai:/root/.shiftai shiftai
# Then point any OpenAI-compatible client at http://localhost:8800/v1

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OLLAMA_HOST=http://host.docker.internal:11434

WORKDIR /app

# Package metadata and source only; the research data and runs stay out of the image.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[server]"

EXPOSE 8800

# 0.0.0.0 so the API is reachable from outside the container.
CMD ["shiftai", "serve", "--host", "0.0.0.0", "--port", "8800"]
