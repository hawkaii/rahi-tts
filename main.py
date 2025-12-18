import io
import re
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

# -------------------------------------------------
# Configuration
# -------------------------------------------------
# T4 GPU works best with FP16. Drastically reduces VRAM and speeds up inference.
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Batching Parameters
# Increased for T4 (16GB VRAM can handle this easily with FP16)
BATCH_SIZE = 8
MAX_WAIT_MS = 50        # Max time to wait for filling a batch
# Max pending chunks before rejecting requests (Backpressure)
QUEUE_SIZE = 100

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("tts")

app = FastAPI(title="Indic Parler-TTS Service")

# -------------------------------------------------
# Global State
# -------------------------------------------------
model = None
tokenizer = None
description_tokenizer = None
tts_queue: asyncio.Queue = None


@dataclass
class TTSJob:
    text: str
    description: str
    future: asyncio.Future = field(default_factory=asyncio.Future)


class TTSRequest(BaseModel):
    text: str
    description: str = "A female speaker delivers a slightly expressive and animated speech with a very clear audio."

# -------------------------------------------------
# Robust Chunking Logic
# -------------------------------------------------


def chunk_text(text: str, max_chars: int = 300) -> List[str]:
    """
    Smart chunking that respects Hindi/Indic sentence endings (Danda).
    """
    # Split by common sentence terminators: ., ?, !, and Hindi Danda (।)
    # The regex keeps the delimiter attached to the sentence.
    sentence_endings = r'(?<=[.!?।])\s+'
    sentences = re.split(sentence_endings, text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # If adding this sentence exceeds max_chars, push current chunk
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk += (" " + sentence) if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence

            # Edge case: If a single sentence is massive, force split it
            while len(current_chunk) > max_chars:
                # Find the nearest space to split
                split_idx = current_chunk.rfind(' ', 0, max_chars)
                if split_idx == -1:
                    split_idx = max_chars

                chunks.append(current_chunk[:split_idx])
                current_chunk = current_chunk[split_idx:].strip()

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

# -------------------------------------------------
# The Batching Engine
# -------------------------------------------------


def run_model_inference(jobs: List[TTSJob]):
    """
    Blocking function to run on a separate thread.
    This prevents the GPU work from freezing the FastAPI event loop.
    """
    try:
        texts = [j.text for j in jobs]
        descriptions = [j.description for j in jobs]

        # Tokenize
        desc_inputs = description_tokenizer(
            descriptions, return_tensors="pt", padding=True
        ).to(DEVICE)

        prompt_inputs = tokenizer(
            texts, return_tensors="pt", padding=True
        ).to(DEVICE)

        # Generate (Blocking GPU Op)
        with torch.no_grad():
            generation = model.generate(
                input_ids=desc_inputs.input_ids,
                attention_mask=desc_inputs.attention_mask,
                prompt_input_ids=prompt_inputs.input_ids,
                prompt_attention_mask=prompt_inputs.attention_mask,
                # Parler-TTS specific generation configs could go here
            )

        # Distribute results back to futures
        for i, job in enumerate(jobs):
            # Move to CPU and numpy
            audio_arr = generation[i].cpu().float().numpy().squeeze()

            # Add a tiny bit of silence (0.1s) to smooth transitions between chunks
            silence = np.zeros(int(model.config.sampling_rate * 0.1))
            final_arr = np.concatenate((audio_arr, silence))

            # Thread-safe way to set result on the loop
            job.future.get_loop().call_soon_threadsafe(job.future.set_result, final_arr)

    except Exception as e:
        logger.error(f"Inference error: {e}")
        for job in jobs:
            if not job.future.done():
                job.future.get_loop().call_soon_threadsafe(job.future.set_exception, e)


async def batch_processor():
    """
    Background task that pulls from queue and sends to the GPU thread.
    """
    logger.info("Worker: Started")
    while True:
        jobs = []
        try:
            # Get first job (blocking wait)
            job = await tts_queue.get()
            jobs.append(job)

            # Smart waiting: wait a bit to see if more jobs arrive to fill the batch
            start_wait = time.time()
            while len(jobs) < BATCH_SIZE:
                elapsed = (time.time() - start_wait) * 1000
                remaining = (MAX_WAIT_MS - elapsed) / 1000.0

                if remaining <= 0:
                    break

                try:
                    # Try to fetch next item with timeout
                    next_job = await asyncio.wait_for(tts_queue.get(), timeout=remaining)
                    jobs.append(next_job)
                except asyncio.TimeoutError:
                    break  # Time's up, process what we have

            # Process the batch
            # CRITICAL: Run blocking inference in a separate thread!
            await asyncio.to_thread(run_model_inference, jobs)

        except Exception as e:
            logger.error(f"Worker loop error: {e}")
        finally:
            # Mark tasks as done in queue
            for _ in jobs:
                tts_queue.task_done()

# -------------------------------------------------
# Lifecycle & API
# -------------------------------------------------


@app.on_event("startup")
async def startup_event():
    global model, tokenizer, description_tokenizer, tts_queue

    logger.info(f"Loading Model on {DEVICE} with {DTYPE}...")

    # Load Model with FP16 for Speed/Memory efficiency
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        "ai4bharat/indic-parler-tts",
        torch_dtype=DTYPE
    ).to(DEVICE)

    tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
    description_tokenizer = AutoTokenizer.from_pretrained(
        model.config.text_encoder._name_or_path)

     # Optimize
     model.eval()
     if torch.cuda.is_available():
         torch.backends.cuda.matmul.allow_tf32 = True
     
     # Log attention mechanism being used
     attn_impl = getattr(model.config, '_attn_implementation', 'default')
     logger.info(f"Using attention implementation: {attn_impl}")

     # Compile for extra speed (try/except in case of version mismatch)
     try:
         model = torch.compile(model, mode="reduce-overhead")
         logger.info("Model compiled successfully")
     except Exception as e:
         logger.warning(f"Could not compile model (safe to ignore): {e}")

    # Initialize Queue and Worker
    tts_queue = asyncio.Queue(maxsize=QUEUE_SIZE)
    asyncio.create_task(batch_processor())
    logger.info("Service Ready 🚀")


@app.post("/generate")
async def generate(req: TTSRequest):
    if tts_queue.full():
        raise HTTPException(
            status_code=503, detail="Server is too busy, try again later.")

    # 1. Chunk the text
    text_chunks = chunk_text(req.text)
    if not text_chunks:
        raise HTTPException(status_code=400, detail="No valid text found.")

    logger.info(f"Received request: {len(text_chunks)} chunks")

    # 2. Create jobs for all chunks
    futures = []
    for chunk in text_chunks:
        fut = asyncio.Future()
        job = TTSJob(text=chunk, description=req.description, future=fut)
        # This might pause if queue is full (Backpressure)
        await tts_queue.put(job)
        futures.append(fut)

    # 3. Wait for all chunks to finish
    try:
        audio_segments = await asyncio.gather(*futures)

        # 4. Stitch audio
        final_audio = np.concatenate(audio_segments)

        # 5. Return WAV
        buffer = io.BytesIO()
        sf.write(buffer, final_audio, model.config.sampling_rate, format="WAV")
        buffer.seek(0)

        return Response(content=buffer.read(), media_type="audio/wav")

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(
            status_code=500, detail="Internal processing error")
