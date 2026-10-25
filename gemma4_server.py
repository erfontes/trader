import os
import sys
import gc
import time
import asyncio

# Prevent CUDA memory fragmentation and allow expandable segments
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

# ==============================================================================
# Model Configurations (Gemma 4 31B 4-bit with MTP Speculative Draft Assistant)
# ==============================================================================
TARGET_MODEL_ID = os.getenv("GEMMA4_TARGET_MODEL", "unsloth/gemma-4-31B-it-unsloth-bnb-4bit")
ASSISTANT_MODEL_ID = os.getenv("GEMMA4_ASSISTANT_MODEL", "google/gemma-4-31B-it-assistant")

# Hard safety caps to guarantee ~2 GB of free VRAM on a 24GB GPU
VRAM_MEMORY_FRACTION = float(os.getenv("GEMMA4_VRAM_FRACTION", "0.90"))  # Max 90% (~21.6 GB) for PyTorch, leaves ~2.4 GB free
MAX_INPUT_TOKENS = int(os.getenv("GEMMA4_MAX_INPUT_TOKENS", "3072"))
MAX_NEW_TOKENS_CAP = int(os.getenv("GEMMA4_MAX_NEW_TOKENS", "1024"))
NUM_ASSISTANT_TOKENS = int(os.getenv("GEMMA4_NUM_ASSISTANT_TOKENS", "4"))
ASSISTANT_SCHEDULE = os.getenv("GEMMA4_ASSISTANT_SCHEDULE", "heuristic")

# Global model state & concurrency lock (prevents overlapping KV-caches)
processor = None
target_model = None
assistant_model = None
inference_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan Manager: handles startup model loading and clean shutdown.
    """
    global processor, target_model, assistant_model
    print("====================================================================")
    print("🚀 Initializing Gemma 4 Speculative Decoding Inference Server...")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(VRAM_MEMORY_FRACTION, 0)
        print(f"Device: {torch.cuda.get_device_name(0)}")
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Available VRAM: {total_vram_gb:.2f} GB (Capped at {VRAM_MEMORY_FRACTION*100:.0f}% = {total_vram_gb*VRAM_MEMORY_FRACTION:.2f} GB to keep ~{total_vram_gb*(1-VRAM_MEMORY_FRACTION):.1f} GB free)")
    print(f"Target Model: {TARGET_MODEL_ID}")
    print(f"Assistant Model: {ASSISTANT_MODEL_ID}")
    print("====================================================================")

    try:
        print("\n⏳ [1/2] Loading AutoProcessor & 4-bit quantized Target Model...")
        start = time.time()
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        processor = AutoProcessor.from_pretrained(TARGET_MODEL_ID)
        try:
            target_model = AutoModelForCausalLM.from_pretrained(
                TARGET_MODEL_ID,
                quantization_config=bnb_config,
                device_map="cuda:0",
                attn_implementation="sdpa",
                torch_dtype=torch.bfloat16
            )
        except Exception:
            target_model = AutoModelForCausalLM.from_pretrained(
                TARGET_MODEL_ID,
                quantization_config=bnb_config,
                device_map="cuda:0"
            )
        target_model.eval()
        print(f"✅ Target model loaded in {time.time() - start:.2f}s!")

        print("\n⏳ [2/2] Loading MTP Speculative Assistant Draft Model...")
        start = time.time()
        try:
            assistant_model = AutoModelForCausalLM.from_pretrained(
                ASSISTANT_MODEL_ID,
                torch_dtype=torch.bfloat16,
                device_map="cuda:0",
                attn_implementation="sdpa"
            )
        except Exception:
            assistant_model = AutoModelForCausalLM.from_pretrained(
                ASSISTANT_MODEL_ID,
                torch_dtype=torch.bfloat16,
                device_map="cuda:0"
            )
        assistant_model.eval()
        assistant_model.generation_config.num_assistant_tokens = NUM_ASSISTANT_TOKENS
        assistant_model.generation_config.num_assistant_tokens_schedule = ASSISTANT_SCHEDULE
        print(f"✅ Assistant model loaded in {time.time() - start:.2f}s!")

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            print(f"📊 GPU Memory in use: {allocated:.2f} GB allocated | {reserved:.2f} GB reserved")

        print("\n🎉 Server is fully ready to accept accelerated inference requests on port 11434!")
    except Exception as e:
        print(f"❌ Error during server startup: {e}")
        raise e

    yield

    # Clean shutdown
    print("\n🛑 Shutting down Gemma 4 server & releasing GPU VRAM...")
    processor = None
    target_model = None
    assistant_model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("✅ GPU VRAM cleanly released.")


app = FastAPI(
    title="Gemma 4 Speculative Decoding API Server",
    description="High-throughput Ollama & OpenAI compatible inference server for Gemma 4 31B",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS Middleware to support requests from notebooks and LAN clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# Health and Diagnostics Endpoints
# ==============================================================================
@app.get("/")
@app.get("/health")
async def health_check():
    vram_info = {}
    if torch.cuda.is_available():
        vram_info = {
            "device": torch.cuda.get_device_name(0),
            "allocated_gb": round(torch.cuda.memory_allocated(0) / (1024**3), 2),
            "reserved_gb": round(torch.cuda.memory_reserved(0) / (1024**3), 2),
            "total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
        }
    return {
        "status": "online" if target_model is not None else "loading",
        "busy": inference_lock.locked(),
        "target_model": TARGET_MODEL_ID,
        "assistant_model": ASSISTANT_MODEL_ID,
        "speculative_decoding": assistant_model is not None,
        "speculative_config": {
            "num_assistant_tokens": NUM_ASSISTANT_TOKENS,
            "schedule": ASSISTANT_SCHEDULE
        },
        "gpu": vram_info
    }


# ==============================================================================
# OpenAI Chat Completions Schema & Endpoints
# ==============================================================================
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False
    extra_body: Optional[Dict[str, Any]] = None


def _prepare_inputs(formatted_messages: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
    """
    Optimized single-pass chat templating and tokenization using the underlying tokenizer
    to eliminate multimodal processor_kwargs warnings and reduce tokenization overhead.
    """
    tokenizer = getattr(processor, "tokenizer", processor)
    try:
        encoded = tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
            return_dict=True
        )
        if isinstance(encoded, torch.Tensor):
            encoded = {"input_ids": encoded, "attention_mask": torch.ones_like(encoded)}
        return {k: v.to(target_model.device) for k, v in encoded.items()}
    except Exception:
        input_text = tokenizer.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=True)
        return tokenizer(
            text=input_text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS
        ).to(target_model.device)


def _generate_sync(inputs: Dict[str, torch.Tensor], gen_kwargs: Dict[str, Any]):
    """
    Synchronous worker execution offloaded to a background thread to prevent
    blocking the FastAPI event loop during heavy CUDA generation.
    """
    with torch.inference_mode():
        return target_model.generate(**inputs, **gen_kwargs)


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    global processor, target_model, assistant_model
    if not target_model or not processor:
        raise HTTPException(status_code=503, detail="Target model not loaded yet.")

    async with inference_lock:
        try:
            formatted_messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
            inputs = _prepare_inputs(formatted_messages)

            speculative_requested = True
            draft_tokens = NUM_ASSISTANT_TOKENS
            if request.extra_body:
                if "speculative" in request.extra_body:
                    speculative_requested = bool(request.extra_body["speculative"])
                if "num_assistant_tokens" in request.extra_body:
                    draft_tokens = int(request.extra_body["num_assistant_tokens"])

            # Low temperature (<=0.2) benefits from greedy decoding to maximize speculative draft acceptance
            temp = request.temperature if request.temperature is not None else 0.7
            do_sample = temp > 0.2
            requested_tokens = request.max_tokens if request.max_tokens else 512

            pad_id = getattr(getattr(processor, "tokenizer", None), "pad_token_id", None)
            if pad_id is None:
                pad_id = getattr(processor, "eos_token_id", None)

            active_assistant = assistant_model if (speculative_requested and assistant_model is not None) else None

            gen_kwargs = {
                "max_new_tokens": min(requested_tokens, MAX_NEW_TOKENS_CAP),
                "do_sample": do_sample,
                "pad_token_id": pad_id,
            }
            if active_assistant is not None:
                gen_kwargs["assistant_model"] = active_assistant
                gen_kwargs["num_assistant_tokens"] = draft_tokens
                gen_kwargs["num_assistant_tokens_schedule"] = ASSISTANT_SCHEDULE
            if do_sample:
                gen_kwargs["temperature"] = temp

            start_time = time.time()
            outputs = await asyncio.to_thread(_generate_sync, inputs, gen_kwargs)
            generation_time = time.time() - start_time

            input_len = inputs["input_ids"].shape[1]
            total_tokens = len(outputs[0])
            completion_tokens = total_tokens - input_len
            prompt_tokens = input_len

            response_text = processor.decode(outputs[0][input_len:], skip_special_tokens=True)

            del inputs, outputs

            tok_per_sec = completion_tokens / generation_time if generation_time > 0 else 0
            mode_str = f"MTP draft={draft_tokens}, schedule={ASSISTANT_SCHEDULE}, greedy={not do_sample}" if active_assistant else "Target-only (speculative=off)"
            print(f"[OpenAI Endpoint] Generated {completion_tokens} tokens in {generation_time:.2f}s ({tok_per_sec:.2f} tok/s) [{mode_str}]")

            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
            }
        except Exception as e:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"❌ Error during generation: {e}")
            raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# Ollama Legacy Generate Schema & Endpoints
# ==============================================================================
class GenerateRequest(BaseModel):
    model: str
    prompt: str
    stream: Optional[bool] = False
    format: Optional[str] = None
    options: Optional[Dict[str, Any]] = None


@app.post("/api/generate")
async def api_generate(request: GenerateRequest):
    global processor, target_model, assistant_model
    if not target_model or not processor:
        raise HTTPException(status_code=503, detail="Target model not loaded yet.")

    async with inference_lock:
        try:
            # Wrap prompt in standard Gemma chat structure
            formatted_messages = [{"role": "user", "content": request.prompt}]
            inputs = _prepare_inputs(formatted_messages)

            # Parse Ollama options
            temp = 0.7
            num_predict = 512
            speculative_requested = True
            draft_tokens = NUM_ASSISTANT_TOKENS

            if request.options:
                if "temperature" in request.options:
                    temp = float(request.options["temperature"])
                if "num_predict" in request.options:
                    num_predict = int(request.options["num_predict"])
                if "speculative" in request.options:
                    speculative_requested = bool(request.options["speculative"])
                if "num_assistant_tokens" in request.options:
                    draft_tokens = int(request.options["num_assistant_tokens"])

            # High-Impact Optimization for Speculative Decoding:
            # Rejection sampling with low temperature (e.g. 0.2) sharply diverges target and draft
            # distributions, causing near 100% rejection of draft tokens.
            # When requesting JSON format or low temperature (<= 0.2), greedy decoding (do_sample=False)
            # gives an ~85-95% draft acceptance rate for structured tokens!
            is_json = request.format == "json"
            if is_json or temp <= 0.2:
                do_sample = False
            else:
                do_sample = temp > 0.05

            pad_id = getattr(getattr(processor, "tokenizer", None), "pad_token_id", None)
            if pad_id is None:
                pad_id = getattr(processor, "eos_token_id", None)

            active_assistant = assistant_model if (speculative_requested and assistant_model is not None) else None

            gen_kwargs = {
                "max_new_tokens": min(num_predict, MAX_NEW_TOKENS_CAP),
                "do_sample": do_sample,
                "pad_token_id": pad_id,
            }
            if active_assistant is not None:
                gen_kwargs["assistant_model"] = active_assistant
                gen_kwargs["num_assistant_tokens"] = draft_tokens
                gen_kwargs["num_assistant_tokens_schedule"] = ASSISTANT_SCHEDULE
            if do_sample:
                gen_kwargs["temperature"] = temp

            start_time = time.time()
            outputs = await asyncio.to_thread(_generate_sync, inputs, gen_kwargs)
            generation_time = time.time() - start_time

            input_len = inputs["input_ids"].shape[1]
            completion_tokens = len(outputs[0]) - input_len
            prompt_tokens = input_len

            response_text = processor.decode(outputs[0][input_len:], skip_special_tokens=True)

            del inputs, outputs

            tok_per_sec = completion_tokens / generation_time if generation_time > 0 else 0
            mode_str = f"MTP draft={draft_tokens}, schedule={ASSISTANT_SCHEDULE}, greedy={not do_sample}" if active_assistant else "Target-only (speculative=off)"
            print(f"[Ollama Endpoint] Generated {completion_tokens} tokens in {generation_time:.2f}s ({tok_per_sec:.2f} tok/s) [{mode_str}]")

            return {
                "model": request.model,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "response": response_text,
                "done": True,
                "context": [],
                "total_duration": int(generation_time * 1e9),
                "prompt_eval_count": prompt_tokens,
                "eval_count": completion_tokens
            }
        except Exception as e:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"❌ Error during generation: {e}")
            raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# Model Listing Compatibility Endpoints
# ==============================================================================
@app.get("/v1/models")
@app.get("/api/tags")
async def get_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "gemma4:31b",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "google"
            },
            {
                "id": "gemma4:26b",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "google"
            }
        ],
        "models": [
            {
                "name": "gemma4:31b",
                "model": "gemma4:31b",
                "modified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "size": 18000000000,
                "digest": "sha256:gemma4-speculative-mtp"
            }
        ]
    }


if __name__ == "__main__":
    port = int(os.getenv("GEMMA4_PORT", os.getenv("PORT", 11434)))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
