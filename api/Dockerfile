FROM python:3.10-slim

WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Move to backend folder
WORKDIR /api

# Expose Render port
EXPOSE 10000

# Run FastAPI backend
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]