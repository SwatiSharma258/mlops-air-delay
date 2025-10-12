FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

ENV PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI="file:/app/mlruns"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
