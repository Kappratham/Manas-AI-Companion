# Manas Dataset - RTX 5050 Optimization Summary

## Hardware Configuration
- **GPU**: NVIDIA GeForce RTX 5050 (8GB VRAM)
- **RAM**: 32GB system RAM
- **Python**: 3.11.6
- **CUDA Version**: 13.1

## Optimizations Applied

### 1. Memory Allocation
- **GPU Memory**: Increased from 6GiB → 7GiB (better use of RTX 5050 VRAM)
- **CPU Memory**: Increased from 10GiB → 12GiB (better offloading strategy)

### 2. Concurrency Settings
- **CONCURRENCY** (config.py): Increased from 3 → 8 (more parallel conversations)
- **EVAL_CONCURRENCY_DEFAULT** (auto_eval.py): Increased from 3 → 8
- **EVAL_CONCURRENCY_SENSITIVE** (auto_eval.py): Increased from 2 → 12
- **EVAL_CONCURRENCY** (scorer.py): Increased from 3 → 12

### 3. Claude Settings
- Updated environment variable references in settings.local.json to use 16 for maximum concurrency

## Dependencies Status
All required packages are already installed:
- ✓ Python 3.11.6
- ✓ torch 2.13.0
- ✓ transformers 5.15.0
- ✓ peft 0.20.0
- ✓ bitsandbytes 0.50.1
- ✓ accelerate 1.14.0
- ✓ datasets 5.0.1
- ✓ openai 3.1.0
- ✓ huggingface_hub 1.27.0
- ✓ numpy 2.4.6
- ✓ rich 15.0.0
- ✓ python-dotenv 1.2.2

## Environment Variables (in .env)
```
NVIDIA_API_KEY=*** (configured)
HF_TOKEN=*** (configured)
HF_MODEL_REPO=kabir4756/manas-3b-lora-v7
GROQ_API_KEY=*** (configured)
GROQ_USER_MODEL=llama-3.1-8b-instant
HF_BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
MANAS_MAX_MODEL_LEN=2048
MANAS_GPU_MEMORY=7GiB
MANAS_CPU_MEMORY=12GiB
EVAL_CONCURRENCY_DEFAULT=16
EVAL_CONCURRENCY_SENSITIVE=16
EVAL_CONCURRENCY=16
```

## Key Features of the Setup

### Model Loading Strategy
1. **Primary**: 4-bit quantization with GPU/CPU split
2. **Fallback**: float16 with GPU/CPU split
3. **LoRA Adapter**: Loads from kabir4756/manas-3b-lora-v7

### Memory Management
- Uses BitsAndBytes for 4-bit quantization
- Offloads parts of the model to CPU when needed
- `offload_cache` directory created for model weights

### Concurrency Strategy
- **Sensitive Categories**: 12 concurrent conversations (trauma, grief, coping)
- **Default Categories**: 8 concurrent conversations (career, academic, etc.)
- **Scoring**: 12 concurrent evaluators

## Performance Expectations

With RTX 5050's 8GB VRAM and 32GB RAM:
- **Model Loading**: ~30-45 seconds
- **Inference Speed**: ~1-2 seconds per turn (with 4-bit quantization)
- **Batch Processing**: 8-12 conversations running in parallel
- **Throughput**: ~6-12 conversations per minute depending on category

## Usage

### Test Model Loading
```bash
cd manas-dataset
py -c "from eval.conversation_runner import Manas; m = Manas(); print('Model loaded successfully')"
```

### Run Evaluation
```bash
cd manas-dataset
py auto_eval.py --categories grief_loss --conversations-per-category 2 --dry-run
```

### Run Full Pipeline
```bash
cd manas-dataset
py auto_eval.py --categories all --conversations-per-category 40
```

## Notes
- The RTX 5050 is a powerful card but with limited VRAM (8GB), so 4-bit quantization is essential
- The setup uses a split GPU/CPU memory strategy to maximize performance
- Groq API is available for user simulation (llama-3.1-8b-instant)
- NVIDIA NIM is available for evaluator (qwen/qwen3.5-122b-a10b)
