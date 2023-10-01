# Use an official Python runtime as the base image
FROM python:3.11.5-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory
COPY . .

# Expose the port that the API will listen on
EXPOSE 8888

# Define the command to run the API when the container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8888"]
