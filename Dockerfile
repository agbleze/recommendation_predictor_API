FROM python:3.10.14-slim
WORKDIR /app

# Install system dependencies (optional, but safe)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your package code and install it
#COPY setup.py setup.cfg reviewoler/ ./
RUN pip install .

# Expose the port your app runs on
EXPOSE 8080

# Run the app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "reviewoler.api.api:app"]
