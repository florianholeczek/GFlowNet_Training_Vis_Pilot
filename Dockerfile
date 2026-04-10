FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY /grid ./grid
COPY dashboard.py .
COPY main.py .
COPY plot_utils.py .

EXPOSE 8050

CMD ["gunicorn", "main:server", "--bind", "0.0.0.0:8050", "--workers", "2"]