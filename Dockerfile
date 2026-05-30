FROM python:3.11

WORKDIR /app

# Upgrade pip tools

RUN pip install --upgrade pip setuptools wheel

# Copy requirements

COPY requirements.txt .

# Install dependencies

RUN pip install --no-cache-dir -r requirements.txt

# Copy project

COPY . .

# Expose FastAPI port

EXPOSE 8000

# Start FastAPI

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
