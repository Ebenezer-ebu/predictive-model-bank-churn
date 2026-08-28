# Dockerfile

FROM python:3.9-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/app/ ./src/app/
COPY models/ ./models/
COPY data/processed/ ./data/processed/

# Create logs directory
RUN mkdir -p logs

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]