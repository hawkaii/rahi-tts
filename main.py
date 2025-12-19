import io
import re
import asyncio
import logging
import time
import os
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

# -------------------------------------------------
# Configuration
# -------------------------------------------------
# T4 GPU works best with FP16. Drastically reduces VRAM and speeds up inference.
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Attention implementation: "flash_attention_2", "sdpa", or "default"
# Can be overridden via ATTN_IMPLEMENTATION environment variable
ATTN_IMPLEMENTATION = os.getenv("ATTN_IMPLEMENTATION", "flash_attention_2")

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


def chunk_text(text: str, max_chars: int = 300, streaming_mode: bool = False) -> List[str]:
    """
    Smart chunking that respects Hindi/Indic sentence endings (Danda).
    
    Args:
        text: Input text to chunk
        max_chars: Maximum characters per chunk
        streaming_mode: If True, uses smaller chunks (50-80 chars) for faster streaming
    """
    # For streaming, use much smaller chunks for instant playback
    if streaming_mode:
        max_chars = 80  # Small chunks = fast generation = instant streaming
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
# Audio Generation Utilities
# -------------------------------------------------


def calculate_max_new_tokens(text: str, frame_rate: int = 86, chars_per_second: int = 10) -> int:
    """
    Calculate appropriate max_new_tokens based on text length.
    This prevents generating excessive trailing silence.
    
    Args:
        text: Input text to be synthesized
        frame_rate: Model's audio frame rate (tokens per second). Default 86 for Parler TTS.
        chars_per_second: Estimated speaking rate in characters per second. 
                         Default 10 is conservative for Hindi/Indic languages.
    
    Returns:
        max_new_tokens: Maximum number of audio tokens to generate
    """
    # Estimate speech duration based on text length
    estimated_duration_sec = len(text) / chars_per_second
    
    # Add 20% buffer for natural pauses and variation in speaking speed
    buffered_duration_sec = estimated_duration_sec * 1.2
    
    # Add minimum 1 second safety margin
    total_duration_sec = buffered_duration_sec + 1.0
    
    # Convert to tokens
    max_new_tokens = int(total_duration_sec * frame_rate)
    
    # Enforce reasonable bounds
    # Minimum: 100 tokens (~1.2 seconds) - enough for very short text
    # Maximum: 2580 tokens (~30 seconds) - model's default max
    max_new_tokens = max(100, min(max_new_tokens, 2580))
    
    return max_new_tokens


# -------------------------------------------------
# Model Warm-up
# -------------------------------------------------


def warmup_model():
    """
    Perform warm-up inference to trigger torch.compile compilation.
    This prevents the first customer request from being extremely slow.
    """
    logger.info("🔥 Starting model warm-up for torch.compile...")
    start_time = time.time()
    
    try:
        # Create dummy inputs
        dummy_text = "यह एक परीक्षण वाक्य है।"  # Hindi test sentence
        dummy_description = "A female speaker delivers a slightly expressive and animated speech with a very clear audio."
        
        # Tokenize
        desc_inputs = description_tokenizer(
            [dummy_description], return_tensors="pt", padding=True
        ).to(DEVICE)
        
        prompt_inputs = tokenizer(
            [dummy_text], return_tensors="pt", padding=True
        ).to(DEVICE)
        
        # Calculate appropriate max_new_tokens for warmup
        warmup_max_tokens = calculate_max_new_tokens(dummy_text)
        
        # Run inference to trigger compilation
        with torch.no_grad():
            _ = model.generate(
                input_ids=desc_inputs.input_ids,
                attention_mask=desc_inputs.attention_mask,
                prompt_input_ids=prompt_inputs.input_ids,
                prompt_attention_mask=prompt_inputs.attention_mask,
                max_new_tokens=warmup_max_tokens,
                pad_token_id=1024,
            )
        
        warmup_time = time.time() - start_time
        logger.info(f"✓ Model warm-up complete in {warmup_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Warm-up failed: {e}")
        logger.warning("Model will still work but first request may be slow")

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

        # Calculate max_new_tokens for the batch (use maximum from all texts)
        # This ensures all texts in the batch have sufficient generation length
        max_tokens_per_text = [calculate_max_new_tokens(text) for text in texts]
        batch_max_new_tokens = max(max_tokens_per_text)
        
        logger.debug(f"Batch of {len(texts)} texts, max_new_tokens={batch_max_new_tokens}")

        # Tokenize
        desc_inputs = description_tokenizer(
            descriptions, return_tensors="pt", padding=True
        ).to(DEVICE)

        prompt_inputs = tokenizer(
            texts, return_tensors="pt", padding=True
        ).to(DEVICE)

        # Generate (Blocking GPU Op) with dynamic max_new_tokens
        with torch.no_grad():
            generation = model.generate(
                input_ids=desc_inputs.input_ids,
                attention_mask=desc_inputs.attention_mask,
                prompt_input_ids=prompt_inputs.input_ids,
                prompt_attention_mask=prompt_inputs.attention_mask,
                max_new_tokens=batch_max_new_tokens,  # Dynamically calculated
                pad_token_id=1024,  # Explicit padding token
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
    logger.info(f"Requested attention implementation: {ATTN_IMPLEMENTATION}")

    # Load Model with FP16 for Speed/Memory efficiency
    # Try to load with requested attention implementation, with graceful fallback
    model_loaded = False
    attn_to_try = [ATTN_IMPLEMENTATION]
    
    # Add fallback options if primary choice fails
    if ATTN_IMPLEMENTATION != "sdpa":
        attn_to_try.append("sdpa")
    if ATTN_IMPLEMENTATION not in ["flash_attention_2", "sdpa"]:
        attn_to_try.append("flash_attention_2")
    
    for attn_impl in attn_to_try:
        try:
            logger.info(f"Attempting to load model with attention: {attn_impl}")
            model = ParlerTTSForConditionalGeneration.from_pretrained(
                "ai4bharat/indic-parler-tts",
                torch_dtype=DTYPE,
                attn_implementation=attn_impl
            ).to(DEVICE)
            model_loaded = True
            logger.info(f"✓ Successfully loaded model with {attn_impl} attention")
            break
        except Exception as e:
            logger.warning(f"Failed to load with {attn_impl}: {e}")
            continue
    
    if not model_loaded:
        # Last resort: load without specifying attention implementation
        try:
            logger.info("Loading model with default attention implementation")
            model = ParlerTTSForConditionalGeneration.from_pretrained(
                "ai4bharat/indic-parler-tts",
                torch_dtype=DTYPE
            ).to(DEVICE)
            logger.info("✓ Loaded model with default attention")
        except Exception as e:
            logger.error(f"Failed to load model with any configuration: {e}")
            raise

    tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
    description_tokenizer = AutoTokenizer.from_pretrained(
        model.config.text_encoder._name_or_path)

    # Optimize
    model.eval()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    
    # Log attention mechanism being used
    attn_impl = getattr(model.config, '_attn_implementation', 'default')
    logger.info(f"Active attention implementation: {attn_impl}")

    # Compile for extra speed with torch.compile
    try:
        # Check if triton is available (required for torch.compile CUDA optimization)
        try:
            import triton
            logger.info(f"✓ Triton available (v{triton.__version__}) - torch.compile will be fast")
        except ImportError:
            logger.warning("⚠️  Triton not found - torch.compile will fall back to eager mode (no speedup)")
        
        logger.info("Compiling model with mode='reduce-overhead'...")
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        logger.info("✓ Model compilation configured (actual compilation happens on first inference)")
    except Exception as e:
        logger.error(f"torch.compile failed: {e}")
        logger.warning("Continuing without compilation")

    # Warm up the model to trigger torch.compile compilation
    await asyncio.to_thread(warmup_model)
    
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


@app.post("/generate-stream")
async def generate_stream(req: TTSRequest):
    """
    Stream audio chunks as they're generated (progressive delivery).
    
    This endpoint provides significantly better perceived latency compared to /generate:
    - First audio chunk plays in ~1 second instead of waiting 5+ seconds
    - Audio chunks stream progressively as they're generated
    - Uses the same efficient batching system as /generate
    
    Response format: multipart audio/wav chunks
    Client should concatenate chunks and play progressively.
    """
    if tts_queue.full():
        raise HTTPException(
            status_code=503, detail="Server is too busy, try again later.")

    # 1. Chunk the text with smaller chunks for faster streaming
    text_chunks = chunk_text(req.text, streaming_mode=True)
    if not text_chunks:
        raise HTTPException(status_code=400, detail="No valid text found.")

    logger.info(f"Received streaming request: {len(text_chunks)} chunks (streaming mode: small chunks)")

    async def audio_chunk_generator():
        """
        Generator that yields audio chunks as they complete.
        Uses micro-batching to submit multiple chunks but yield as soon as each completes.
        """
        try:
            # 2. Submit ALL chunks to queue immediately (leverages batching)
            futures = []
            for i, chunk in enumerate(text_chunks):
                fut = asyncio.Future()
                job = TTSJob(text=chunk, description=req.description, future=fut)
                await tts_queue.put(job)
                futures.append(fut)
            
            logger.debug(f"Queued {len(text_chunks)} chunks for processing")
            
            # 3. Yield chunks as SOON as they complete (not in order, for fastest response)
            # Use asyncio.wait to get results as they finish
            pending = set(futures)
            completed_count = 0
            
            while pending:
                # Wait for next chunk to complete
                done, pending = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Yield all completed chunks
                for fut in done:
                    completed_count += 1
                    audio_arr = await fut
                    
                    # Convert to WAV format
                    buffer = io.BytesIO()
                    sf.write(buffer, audio_arr, model.config.sampling_rate, format="WAV")
                    buffer.seek(0)
                    wav_data = buffer.read()
                    
                    # Get chunk index
                    chunk_idx = futures.index(fut)
                    
                    # Yield chunk with metadata
                    chunk_header = f"--chunk\r\nContent-Type: audio/wav\r\nX-Chunk-Index: {chunk_idx}\r\nX-Total-Chunks: {len(text_chunks)}\r\n\r\n".encode()
                    yield chunk_header
                    yield wav_data
                    yield b"\r\n"
                    
                    logger.debug(f"Streamed chunk {chunk_idx+1}/{len(text_chunks)} (completed {completed_count}/{len(text_chunks)})")
            
            # Final boundary marker
            yield b"--chunk--\r\n"
            logger.info(f"Completed streaming {len(text_chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            # Send error marker
            error_msg = f"--chunk\r\nContent-Type: text/plain\r\nX-Error: true\r\n\r\nError: {str(e)}\r\n--chunk--\r\n".encode()
            yield error_msg

    return StreamingResponse(
        audio_chunk_generator(),
        media_type="multipart/x-mixed-replace; boundary=chunk",
        headers={
            "X-Content-Type": "audio/wav",
            "X-Streaming": "true",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
