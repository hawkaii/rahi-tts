FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

# Basic tools
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps
# Note: parler-tts requires transformers==4.46.1 (pinned in their setup.py)
# Install flash-attn first as it requires compilation
RUN pip install --no-cache-dir \
    "flash-attn>=2.3.0" \
    "git+https://github.com/huggingface/parler-tts.git" \
    fastapi \
    uvicorn \
    soundfile \
    scipy \
    protobuf \
    sentencepiece

COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
