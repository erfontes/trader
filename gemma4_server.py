import os
import sys
import gc
import time
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

# Global model state
processor = None
target_model = None
assistant_model = None


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
        print(f"Device: {torch.cuda.get_device_name(0)}")
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Available VRAM: {total_vram_gb:.2f} GB")
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
        target_model = AutoModelForCausalLM.from_pretrained(
            TARGET_MODEL_ID,
            quantization_config=bnb_config,
            device_map="cuda:0"
        )
        print(f"✅ Target model loaded in {time.time() - start:.2f}s!")

        print("\n⏳ [2/2] Loading MTP Speculative Assistant Draft Model...")
        start = time.time()
        assistant_model = AutoModelForCausalLM.from_pretrained(
            ASSISTANT_MODEL_ID,
            dtype=torch.bfloat16,
            device_map="cuda:0"
        )
        assistant_model.generation_config.num_assistant_tokens = 4
        assistant_model.generation_config.num_assistant_tokens_schedule = "heuristic"
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
        "target_model": TARGET_MODEL_ID,
        "assistant_model": ASSISTANT_MODEL_ID,
        "speculative_decoding": assistant_model is not None,
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


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    global processor, target_model, assistant_model
    if not target_model or not assistant_model or not processor:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    try:
        formatted_messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        input_text = processor.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=input_text, return_tensors="pt").to(target_model.device)

        do_sample = request.temperature is not None and request.temperature > 0
        gen_kwargs = {
            "max_new_tokens": request.max_tokens if request.max_tokens else 512,
            "do_sample": do_sample,
            "assistant_model": assistant_model,
        }
        if do_sample:
            gen_kwargs["temperature"] = request.temperature

        start_time = time.time()
        with torch.no_grad():
            outputs = target_model.generate(**inputs, **gen_kwargs)
        generation_time = time.time() - start_time

        input_len = inputs["input_ids"].shape[1]
        del inputs
        torch.cuda.empty_cache()

        response_text = processor.decode(outputs[0][input_len:], skip_special_tokens=True)

        prompt_tokens = input_len
        completion_tokens = len(outputs[0]) - input_len
        total_tokens = len(outputs[0])

        tok_per_sec = completion_tokens / generation_time if generation_time > 0 else 0
        print(f"[OpenAI Endpoint] Generated {completion_tokens} tokens in {generation_time:.2f}s ({tok_per_sec:.2f} tok/s)")

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
    if not target_model or not assistant_model or not processor:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    try:
        # Wrap prompt in standard Gemma chat structure
        formatted_messages = [{"role": "user", "content": request.prompt}]
        input_text = processor.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=input_text, return_tensors="pt").to(target_model.device)

        # Parse Ollama options
        temp = 0.7
        num_predict = 512
        if request.options:
            if "temperature" in request.options:
                temp = float(request.options["temperature"])
            if "num_predict" in request.options:
                num_predict = int(request.options["num_predict"])

        do_sample = temp > 0
        gen_kwargs = {
            "max_new_tokens": num_predict,
            "do_sample": do_sample,
            "assistant_model": assistant_model,
        }
        if do_sample:
            gen_kwargs["temperature"] = temp

        start_time = time.time()
        with torch.no_grad():
            outputs = target_model.generate(**inputs, **gen_kwargs)
        generation_time = time.time() - start_time

        input_len = inputs["input_ids"].shape[1]
        del inputs
        torch.cuda.empty_cache()

        response_text = processor.decode(outputs[0][input_len:], skip_special_tokens=True)

        prompt_tokens = input_len
        completion_tokens = len(outputs[0]) - input_len

        tok_per_sec = completion_tokens / generation_time if generation_time > 0 else 0
        print(f"[Ollama Endpoint] Generated {completion_tokens} tokens in {generation_time:.2f}s ({tok_per_sec:.2f} tok/s)")

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
