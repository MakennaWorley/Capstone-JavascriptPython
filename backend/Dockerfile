FROM python:3.12
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend /app
ENV CORS_ORIGINS="http://localhost:5173,http://localhost:8501"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
