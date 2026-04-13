# Use the python:3.12-slim image as a base
FROM python:3.12-slim

# Set the working directory to /app
WORKDIR /app

# Copy the requirement.txt file to the work diretory
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the files
COPY . .

# Health check for AWS App Runner
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860')"

# Expose port 7860 which the default port for Gradio;
# the EXPOSE instruction only functions as documentation
EXPOSE 7860

ENV GRADIO_SERVER_NAME="0.0.0.0"

# Run app.py
CMD ["python", "main.py"]