FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1.py310:latest

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./

ENTRYPOINT ["python", "-m", "gcp_task"]