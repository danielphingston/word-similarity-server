# Use official Python image
FROM python:3.12-slim

# Set work directory
WORKDIR /app

# Install OS dependencies (for joblib, annoy, etc.)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy requirements if you have one (optional but preferred)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your entire app
COPY . .

# Expose port (Railway sets PORT env var)
EXPOSE 8000

# Start with hypercorn
CMD ["hypercorn", "main:app", "--bind", "[::]:8000"]
