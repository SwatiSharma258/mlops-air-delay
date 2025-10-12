# ================================================
# Dockerfile for Flight Delay Prediction API
# ================================================

# 1️⃣ Use official Python base image
FROM python:3.10-slim

# 2️⃣ Set working directory
WORKDIR /app

# 3️⃣ Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# 4️⃣ Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5️⃣ Copy the rest of the project
COPY . .

# 6️⃣ Expose FastAPI default port
EXPOSE 8000

# 7️⃣ Set environment variables
ENV PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI="file:/app/mlruns"

# 8️⃣ Start FastAPI app using uvicorn
#    --reload can be added for dev purposes
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
