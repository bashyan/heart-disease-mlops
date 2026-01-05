# Use official lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# 1. Install Dependencies
# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy Source Code
COPY app/ app/
COPY src/ src/

# The application now fetches the model securely from DagsHub ML Flow Registry at runtime.

# Create logs directory
RUN mkdir -p logs

# 4. Expose Port
EXPOSE 8000

# 5. Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# 6. Run the Application
# We use host 0.0.0.0 to allow external access
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]