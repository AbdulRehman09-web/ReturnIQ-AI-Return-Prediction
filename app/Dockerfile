FROM python:3.10-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# # Move to frontend folder
# WORKDIR /app

# Expose Render port
EXPOSE 10000

# Run Streamlit app
CMD ["sh", "-c", "streamlit run app.py --server.port $PORT --server.address 0.0.0.0"]