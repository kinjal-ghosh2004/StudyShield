FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files and (eventually) model weights
COPY ./deployment /app/deployment

# Expose port for FastAPI
EXPOSE 8000

# Start server using Uvicorn
CMD ["uvicorn", "deployment.api:app", "--host", "0.0.0.0", "--port", "8000"]
