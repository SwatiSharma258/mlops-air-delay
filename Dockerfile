# -------------------------------
# Dockerfile for Flight Delay Prediction API
# -------------------------------

# 1️⃣ Use an official Python base image
FROM python:3.10-slim

# 2️⃣ Set working directory
WORKDIR /app

# 3️⃣ Copy everything from your project into the container
COPY . /app

# 4️⃣ Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5️⃣ Expose FastAPI default port
EXPOSE 8000

# 6️⃣ Start FastAPI app using uvicorn
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
