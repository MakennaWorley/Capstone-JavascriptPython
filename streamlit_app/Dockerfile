FROM python:3.12-slim
WORKDIR /app
COPY streamlit_app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY streamlit_app /app
ENV STREAMLIT_SERVER_PORT=8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
