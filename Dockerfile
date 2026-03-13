FROM python:3.13-slim

RUN apt-get update && apt-get install -y libpq-dev gcc git

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .
