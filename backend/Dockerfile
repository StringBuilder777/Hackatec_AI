FROM python:3.12-slim

WORKDIR /app

# Sistema: curl para healthcheck, build deps mínimos
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Dependencias Python
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Código backend
COPY backend/ /app/

# Crear directorios persistentes
RUN mkdir -p /app/data /app/skills/generated /app/frontend

EXPOSE 8080

# Hot reload en desarrollo (volumes lo sobreescribe igualmente)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
