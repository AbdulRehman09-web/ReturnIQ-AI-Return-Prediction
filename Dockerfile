FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Render provides PORT dynamically
ENV PORT=10000

# Start FastAPI with uvicorn
CMD uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1