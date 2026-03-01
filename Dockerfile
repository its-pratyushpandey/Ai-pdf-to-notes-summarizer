FROM python:3.13-slim

WORKDIR /app

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY . /app

ENV PYTHONUNBUFFERED=1
EXPOSE 8001

CMD ["python", "-m", "uvicorn", "ai_summarizer.api.app:app", "--host", "0.0.0.0", "--port", "8001"]
