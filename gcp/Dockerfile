# Usamos la imagen oficial de PyTorch con CUDA 12.1 (Compatible con L4)
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Evita que Python genere archivos .pyc y buffer de salida (logs más rápidos en Cloud)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Evita conflictos con tokenizers en entornos con múltiples núcleos y num_workers=2 o más
ENV TOKENIZERS_PARALLELISM=false

WORKDIR /app

# Instalamos dependencias de sistema si fueran necesarias (opcional pero recomendado)
# git a veces es necesario si transformers necesita bajar algo específico, o procps para monitoreo
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiamos y instalamos requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiamos el código
COPY *.py ./

# Ejecutamos
ENTRYPOINT ["python", "-m", "gcp_task"]