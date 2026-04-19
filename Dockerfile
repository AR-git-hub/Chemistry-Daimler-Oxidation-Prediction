FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app/src

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "scripts/predict.py", "--output-path", "predictions.csv"]

