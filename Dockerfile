# Use slim python base for smaller size
FROM python:3.11-slim

# 1. Install system dependencies (these rarely change)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# 2. Copy ONLY the requirements file first
COPY requirements.in .

RUN pip install --upgrade pip pip-tools && \
    pip-compile \
        --no-emit-index-url \
        --no-emit-options \
        --generate-hashes \
        requirements.in \
        -o requirement2.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=100 --retries=5 --no-cache-dir \
    -r requirement2.txt

# 4. Copy ONLY the relevant application files.
# This creates a lean image and a robust build cache.
COPY yolo_jp_product_flask_v1.py .
COPY crnn_model_jpy_valloss0.2_10k.keras .
COPY date_telephone_jp_best.pt .
COPY text_chunk_epoch40_best.pt .
COPY IMG_0942_.jpg .

# 5. Create the output directory
RUN mkdir output

# Run your script by default
ENTRYPOINT ["python", "yolo_jp_product_flask_v1.py"]