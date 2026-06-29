FROM python:3.10-slim

WORKDIR /code

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better Docker caching)
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project
COPY . .

# Hugging Face Space port
EXPOSE 7860

# Disable Streamlit telemetry
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Launch Streamlit
CMD ["streamlit", "run", "deployment/app.py", "--server.port=7860", "--server.address=0.0.0.0"]