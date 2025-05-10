# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . /app/

# Expose the port your app runs on (e.g., 8000 for FastAPI)
EXPOSE 8000

# Run the app (adjust this if you use something else)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
