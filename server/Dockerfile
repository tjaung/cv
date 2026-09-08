FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app

# OpenCV runtime libraries, without a desktop environment.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY server/requirements.txt ./server/requirements.txt
# CPU wheels avoid installing CUDA libraries on a local Docker setup.
RUN python -m pip install 'torch>=2.6,<3' 'torchvision>=0.21,<1' --index-url https://download.pytorch.org/whl/cpu
COPY CV ./CV
RUN python -m pip install '.[cnn]' -r server/requirements.txt
COPY server ./server
RUN python -c "from server.main import app; app.openapi()"

EXPOSE 8000
# One worker keeps the model cache and background job registry in one process.
CMD ["python", "-m", "uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
