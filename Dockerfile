FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# (Optionnal but recommanded) non-root user
RUN useradd -m appuser

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install -U pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# Code
COPY . /app

# Port (informative)
EXPOSE 7860

USER appuser

CMD ["python", "app.py"]
