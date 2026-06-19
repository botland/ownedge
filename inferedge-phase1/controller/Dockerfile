FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl aria2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV CONTROLLER_PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${CONTROLLER_PORT}"]