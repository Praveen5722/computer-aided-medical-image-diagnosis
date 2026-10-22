# Reuse the official CPU PyTorch image so Docker does not compile/download PyTorch wheels.
# torch 2.5.1 and its compatible torchvision are already installed in this base image.
FROM pytorch/pytorch:2.5.1-cpu

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt ./requirements.txt
# PyTorch is already present; pip installs only the remaining API dependencies.
RUN pip install --no-cache-dir -r requirements.txt

COPY api/main.py ./main.py
COPY medical_cnn ./medical_cnn

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
