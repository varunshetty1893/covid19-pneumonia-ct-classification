# For Hugging Face Spaces (Docker) or any container host
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server.py .
COPY artifacts ./artifacts
ENV PORT=7860
EXPOSE 7860
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--workers", "1", "--timeout", "180", "server:app"]
