"""Generate manas_finetune.ipynb programmatically."""
import json
from pathlib import Path

MANAS_SYSTEM = """You are Manas — a mental health companion, not a therapist. You listen deeply,
ask precise questions, and structure the user's thought process gently — helping
things feel lighter without telling them how to feel.

You are warm but not soft. Honest but not harsh. Cynical enough to see through
performance, grounded enough not to be swept away by what you see. You have your
own center of gravity. You are not reactive — you do not simply mirror the user
back at them. You notice what is underneath.

For the first 3 to 4 turns you are in silent assessment — reading tone, language
patterns, and trigger words to understand what is actually happening beneath the
surface. From turn 4 onwards you gently introduce structure, framing insights as
lived observation rather than analysis.

You hold all wisdom traditions as equal tools — Buddhist acceptance, Stoic clarity,
Frankl's meaning-making, Adlerian social psychology, existentialist freedom,
Nietzsche's self-overcoming, depth psychology, attachment theory, polyvagal theory,
CBT frameworks, IFS. You hold no allegiance to any one. You reach for whichever
lens creates the most clarity for this specific person in this specific moment.

You introduce frameworks in soft surface mode by default — insight without naming
the theory. If the user shows curiosity, you name it and go deeper. One framework
per turn. If they do not engage, you drop it and move with them.

You follow a conversation arc: validate first, explore second, surface what is
underneath, offer one small specific suggestion only when the person genuinely
feels heard. You ask one open question per turn — never two. You mirror the
user's language — English or Hinglish or regional mix, whatever they bring.

You never diagnose. You never give medical advice. You never suggest big life
changes. You never shame a coping mechanism — understand its function before
offering an alternative. You never say you should have asked for help sooner.
You never build dependency — your goal is to help the person's internal voice
get louder, not to replace it.

For acute crisis signals — active self-harm, specific suicidal ideation, immediate
danger — acknowledge warmly, provide iCall 9152987821 and Vandrevala Foundation
1860-2662-345 immediately, and stay present. You do not abandon.

You are transparent that you are AI. You are the external voice that exists to
make the internal voice clearer."""


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [text + "\n"]
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [source + "\n"]
    }


TITLE = """# Manas Fine-Tuning — QLoRA on Qwen2.5 14B
Platform: Google Colab (Free NVIDIA T4 16GB)
Method: QLoRA via Unsloth
Dataset: kabir4756/manas-dataset (private)

**Before running:**
1. Runtime → Change runtime type → GPU T4 → Save
2. Runtime → Run all (or run cells with Shift+Enter)
3. Your HF token is already configured in Cell 2 — no setup needed
4. Expected runtime: 8-12 hours for 400 steps
5. Keep the tab open — Colab disconnects after ~90 min of inactivity
   (Colab Pro extends session duration significantly)
"""

cells = [
    md(TITLE),

    # Cell 1
    md("## Cell 1 — Install Dependencies"),
    code("%%capture\n"
         "!pip install unsloth\n"
         "!pip install trl peft accelerate bitsandbytes\n"
         "!pip install datasets huggingface_hub\n"
         "!pip install torch torchvision --upgrade\n"
         "# Use ModelScope fallback if HF is unreachable\n"
         "!pip install modelscope\n"
         "import os; os.environ['UNSLOTH_USE_MODELSCOPE'] = '1'"),

    # Cell 2
    md("## Cell 2 — Imports and Secrets"),
    code("# ============================================================\n"
         "# HF TOKEN — Replace the placeholder with your actual token\n"
         "# Get it from: https://huggingface.co/settings/tokens\n"
         "# ============================================================\n"
         "HF_TOKEN = \"YOUR_HF_TOKEN_HERE\"\n"
         "\n"
         "import os\n"
         "import math\n"
         "import torch\n"
         "\n"
         "from unsloth import FastLanguageModel\n"
         "from unsloth.chat_templates import get_chat_template\n"
         "\n"
         "from datasets import load_dataset\n"
         "from trl import SFTTrainer\n"
         "from transformers import TrainingArguments, DataCollatorForSeq2Seq\n"
         "from huggingface_hub import HfApi, login\n"
         "\n"
         "login(token=HF_TOKEN)\n"
         "\n"
         "print(f\"GPU count: {torch.cuda.device_count()}\")\n"
         "print(f\"GPU 0: {torch.cuda.get_device_name(0)}\")\n"
         "if torch.cuda.device_count() > 1:\n"
         "    print(f\"GPU 1: {torch.cuda.get_device_name(1)}\")\n"
         "print(f\"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB x{torch.cuda.device_count()}\")"),

    # Cell 3
    md("## Cell 3 — Config"),
    code("CONFIG = {\n"
         "    # Model\n"
         "    \"base_model\": \"unsloth/Qwen2.5-14B-Instruct-bnb-4bit\",\n"
         "    \"max_seq_length\": 4096,\n"
         "    \"dtype\": None,  # auto-detect\n"
         "    \"load_in_4bit\": True,\n"
         "\n"
         "    # LoRA\n"
         "    \"lora_r\": 16,\n"
         "    \"lora_alpha\": 32,\n"
         "    \"lora_dropout\": 0,\n"
         "    \"target_modules\": [\n"
         "        \"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\",\n"
         "        \"gate_proj\", \"up_proj\", \"down_proj\"\n"
         "    ],\n"
         "\n"
         "    # Training\n"
         "    \"batch_size\": 1,\n"
         "    \"gradient_accumulation_steps\": 8,\n"
         "    \"warmup_steps\": 10,\n"
         "    \"max_steps\": 400,\n"
         "    \"learning_rate\": 2e-4,\n"
         "    \"weight_decay\": 0.01,\n"
         "    \"lr_scheduler_type\": \"cosine\",\n"
         "    \"seed\": 42,\n"
         "\n"
         "    # Checkpointing\n"
         "    \"save_steps\": 100,\n"
         "    \"output_dir\": \"/content/manas-checkpoints\",\n"
         "\n"
         "    # Dataset\n"
         "    \"hf_dataset_repo\": \"kabir4756/manas-dataset\",\n"
         "\n"
         "    # HF push\n"
         "    \"hf_model_repo\": \"kabir4756/manas-14b-lora\",\n"
         "}\n"
         "\n"
         "print(\"CONFIG loaded. Key settings:\")\n"
         "print(f\"  base_model: {CONFIG['base_model']}\")\n"
         "print(f\"  max_seq_length: {CONFIG['max_seq_length']}, lora_r: {CONFIG['lora_r']}\")\n"
         "print(f\"  max_steps: {CONFIG['max_steps']}, lr: {CONFIG['learning_rate']}\")\n"
         "print(f\"  dataset: {CONFIG['hf_dataset_repo']}\")\n"
         "print(f\"  model push: {CONFIG['hf_model_repo']}\")"),

    # Cell 4
    md("## Cell 4 — Load Dataset from HF Hub"),
    code("dataset = load_dataset(CONFIG[\"hf_dataset_repo\"], token=HF_TOKEN)\n"
         "\n"
         "train_data = dataset[\"train\"]\n"
         "eval_data = dataset[\"eval\"]\n"
         "\n"
         "print(f\"Train conversations: {len(train_data)}\")\n"
         "print(f\"Eval conversations: {len(eval_data)}\")\n"
         "\n"
         "# Preview first train sample\n"
         "sample = train_data[0]\n"
         "msgs = sample.get('messages', [])\n"
         "system_turn = [m for m in msgs if m['role'] == 'system']\n"
         "user_turns  = [m for m in msgs if m['role'] == 'user']\n"
         "asst_turns  = [m for m in msgs if m['role'] == 'assistant']\n"
         "print(\"\\nFirst train sample:\")\n"
         "print(\"  System turn : \" + (\"Yes\" if system_turn else \"No\"))\n"
         "print(\"  User turns  : \" + str(len(user_turns)))\n"
         "print(\"  Assistant   : \" + str(len(asst_turns)))\n"
         "print(\"\\n  First user message (trimmed):\")\n"
         "first_user = next((m for m in msgs if m['role'] == 'user'), None)\n"
         "if first_user:\n"
         "    print(\"  \" + first_user['content'][:300] + \"...\")\n"
         "else:\n"
         "    print(\"  [No user message found]\")\n"
         "print(\"\\n  Second message:\")\n"
         "if len(msgs) > 1:\n"
         "    print(\"  \" + msgs[1]['role'] + \": \" + msgs[1]['content'][:200] + \"...\")\n"
         "else:\n"
         "    print(\"  [Only system message]\")\n"),

    # Cell 5
    md("## Cell 5 — Load Model and Tokenizer"),
    code("model, tokenizer = FastLanguageModel.from_pretrained(\n"
         "    model_name=CONFIG[\"base_model\"],\n"
         "    max_seq_length=CONFIG[\"max_seq_length\"],\n"
         "    dtype=CONFIG[\"dtype\"],\n"
         "    load_in_4bit=CONFIG[\"load_in_4bit\"],\n"
         "    token=HF_TOKEN,\n"
         ")\n"
         "\n"
         "# Apply Qwen2.5 chat template\n"
         "tokenizer = get_chat_template(tokenizer, chat_template=\"qwen-2.5\")\n"
         "\n"
         "print(f\"Model: {CONFIG['base_model']}\")\n"
         "print(f\"Parameters: {model.num_parameters() / 1e9:.1f}B ({'4-bit quantized' if CONFIG['load_in_4bit'] else 'fp16'})\")"),

    # Cell 6
    md("## Cell 6 — Apply QLoRA with Unsloth"),
    code("model = FastLanguageModel.get_peft_model(\n"
         "    model,\n"
         "    r=CONFIG[\"lora_r\"],\n"
         "    target_modules=CONFIG[\"target_modules\"],\n"
         "    lora_alpha=CONFIG[\"lora_alpha\"],\n"
         "    lora_dropout=CONFIG[\"lora_dropout\"],\n"
         "    bias=\"none\",\n"
         "    use_gradient_checkpointing=\"unsloth\",\n"
         "    random_state=CONFIG[\"seed\"],\n"
         "    use_rslora=False,\n"
         "    loftq_config=None,\n"
         ")\n"
         "\n"
         "model.print_trainable_parameters()"),

    # Cell 7
    md("## Cell 7 — Format Dataset into Training Format"),
    code("def format_conversation(example):\n"
         "    \"\"\"Apply Qwen chat template to each conversation.\"\"\"\n"
         "    messages = example[\"messages\"]\n"
         "    text = tokenizer.apply_chat_template(\n"
         "        messages,\n"
         "        tokenize=False,\n"
         "        add_generation_prompt=False\n"
         "    )\n"
         "    return {\"text\": text}\n"
         "\n"
         "train_formatted = train_data.map(\n"
         "    format_conversation,\n"
         "    remove_columns=train_data.column_names\n"
         ")\n"
         "eval_formatted = eval_data.map(\n"
         "    format_conversation,\n"
         "    remove_columns=eval_data.column_names\n"
         ")\n"
         "\n"
         "print(\"Sample formatted text (first 600 chars):\")\n"
         "print(train_formatted[0][\"text\"][:600])\n"
         "print(\"\\n---\")\n"
         "print(f\"Train: {len(train_formatted)} | Eval: {len(eval_formatted)}\")"),

    # Cell 8
    md("## Cell 8 — Training"),
    code("trainer = SFTTrainer(\n"
         "    model=model,\n"
         "    tokenizer=tokenizer,\n"
         "    train_dataset=train_formatted,\n"
         "    eval_dataset=eval_formatted,\n"
         "    dataset_text_field=\"text\",\n"
         "    max_seq_length=CONFIG[\"max_seq_length\"],\n"
         "    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),\n"
         "    dataset_num_proc=2,\n"
         "    packing=False,  # Variable-length convos — no packing\n"
         "    args=TrainingArguments(\n"
         "        per_device_train_batch_size=CONFIG[\"batch_size\"],\n"
         "        gradient_accumulation_steps=CONFIG[\"gradient_accumulation_steps\"],\n"
         "        warmup_steps=CONFIG[\"warmup_steps\"],\n"
         "        max_steps=CONFIG[\"max_steps\"],\n"
         "        learning_rate=CONFIG[\"learning_rate\"],\n"
         "        fp16=not torch.cuda.is_bf16_supported(),\n"
         "        bf16=torch.cuda.is_bf16_supported(),\n"
         "        logging_steps=10,\n"
         "        optim=\"adamw_8bit\",\n"
         "        weight_decay=CONFIG[\"weight_decay\"],\n"
         "        lr_scheduler_type=CONFIG[\"lr_scheduler_type\"],\n"
         "        seed=CONFIG[\"seed\"],\n"
         "        output_dir=CONFIG[\"output_dir\"],\n"
         "        save_steps=CONFIG[\"save_steps\"],\n"
         "        save_total_limit=3,\n"
         "        evaluation_strategy=\"steps\",\n"
         "        eval_steps=100,\n"
         "        load_best_model_at_end=True,  # Push best checkpoint, not last step\n"
         "        report_to=\"none\",\n"
         "        dataloader_pin_memory=False,  # Avoid memory conflicts on 2x T4\n"
         "    ),\n"
         ")\n"
         "\n"
         "gpu_stats = torch.cuda.get_device_properties(0)\n"
         "start_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 3)\n"
         "max_mem = round(gpu_stats.total_memory / 1024**3, 3)\n"
         "print(f\"GPU: {gpu_stats.name} | Memory: {max_mem}GB x{torch.cuda.device_count()}\")\n"
         "print(f\"Reserved before training: {start_mem}GB\")\n"
         "print(f\"\\nStarting training... (~8-12 hours for 400 steps)\")\n"
         "print(f\"Effective batch size: {CONFIG['batch_size'] * CONFIG['gradient_accumulation_steps']} | Total VRAM needed: ~{max_mem * torch.cuda.device_count():.0f}GB\")\n"
         "trainer_stats = trainer.train()\n"
         "\n"
         "used_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 3)\n"
         "print(f\"\\n{'='*60}\")\n"
         "print(f\"Training complete!\")\n"
         "print(f\"Steps: {trainer_stats.metrics['step']}\")\n"
         "print(f\"Train loss: {trainer_stats.metrics['train_loss']:.4f}\")\n"
         "print(f\"Runtime: {trainer_stats.metrics['train_runtime']:.0f}s ({trainer_stats.metrics['train_runtime']/60:.1f} min)\")\n"
         "print(f\"Peak GPU memory: {used_mem}GB\")"),

    # Cell 9
    md("## Cell 9 — Save Checkpoint Locally"),
    code("local_save_path = \"/content/manas-lora-final\"\n"
         "model.save_pretrained(local_save_path)\n"
         "tokenizer.save_pretrained(local_save_path)\n"
         "\n"
         "print(f\"LoRA adapter saved: {local_save_path}\")\n"
         "\n"
         "import os\n"
         "total_size = 0\n"
         "for f in sorted(os.listdir(local_save_path)):\n"
         "    fp = os.path.join(local_save_path, f)\n"
         "    if os.path.isfile(fp):\n"
         "        size = os.path.getsize(fp) / 1024 / 1024\n"
         "        total_size += size\n"
         "        print(f\"  {f}: {size:.1f}MB\")\n"
         "    elif os.path.isdir(fp):\n"
         "        for sf in sorted(os.listdir(fp)):\n"
         "            sfp = os.path.join(fp, sf)\n"
         "            size = os.path.getsize(sfp) / 1024 / 1024\n"
         "            total_size += size\n"
         "            print(f\"  {f}/{sf}: {size:.1f}MB\")\n"
         "print(f\"\\nTotal saved: {total_size:.1f}MB\")"),

    # Cell 10
    md("## Cell 10 — Evaluate on Eval Split"),
    code(f"MANAS_SYSTEM_PROMPT = '''{MANAS_SYSTEM}'''\n"
         "\n"
         "# Quantitative\n"
         "print(\"Running quantitative eval...\")\n"
         "eval_results = trainer.evaluate()\n"
         "print(f\"Eval loss: {eval_results['eval_loss']:.4f}\")\n"
         "print(f\"Eval perplexity: {math.exp(eval_results['eval_loss']):.2f}\")\n"
         "\n"
         "# Qualitative\n"
         "FastLanguageModel.for_inference(model)\n"
         "\n"
         "QUALITATIVE_SAMPLES = [\n"
         "    \"yaar bohot thaka hua hoon aajkal, kuch achha nahi lag raha\",\n"
         "    \"I've been feeling really lost lately. I don't know what I want anymore.\",\n"
         "    \"Mujhe lagta hai main apne parents ko disappoint kar raha hoon har cheez mein\",\n"
         "    \"I lost my mom 3 months ago. I'm fine, I think. Back at work and everything.\",\n"
         "    \"I know exactly what my problem is. I've read about it. I just can't stop doing it.\",\n"
         "]\n"
         "\n"
         "print(\"\\n\" + \"=\"*60)\n"
         "print(\"QUALITATIVE EVALUATION — 5 SAMPLES\")\n"
         "print(\"=\"*60)\n"
         "\n"
         "for i, user_msg in enumerate(QUALITATIVE_SAMPLES):\n"
         "    messages = [{\"role\": \"system\", \"content\": MANAS_SYSTEM_PROMPT},\n"
         "                {\"role\": \"user\", \"content\": user_msg}]\n"
         "    inputs = tokenizer.apply_chat_template(\n"
         "        messages, tokenize=True, add_generation_prompt=True, return_tensors=\"pt\"\n"
         "    ).to(\"cuda\")\n"
         "    outputs = model.generate(\n"
         "        input_ids=inputs,\n"
         "        max_new_tokens=300,\n"
         "        temperature=0.75,\n"
         "        do_sample=True,\n"
         "        pad_token_id=tokenizer.eos_token_id,\n"
         "    )\n"
         "    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)\n"
         "    print(f\"\\nSample {i+1}:\")\n"
         "    print(f\"User: {user_msg}\")\n"
         "    print(f\"Manas: {response}\")\n"
         "    print(\"-\"*60)"),

    # Cell 11
    md("## Cell 11 — Push Adapter to HF Hub"),
    code("print(\"Pushing LoRA adapter to HF Hub...\")\n"
         "print(f\"Target: https://huggingface.co/{CONFIG['hf_model_repo']}\\n\")\n"
         "\n"
         "model.push_to_hub(\n"
         "    CONFIG[\"hf_model_repo\"],\n"
         "    token=HF_TOKEN,\n"
         "    private=True\n"
         ")\n"
         "tokenizer.push_to_hub(\n"
         "    CONFIG[\"hf_model_repo\"],\n"
         "    token=HF_TOKEN,\n"
         "    private=True\n"
         ")\n"
         "\n"
         "api = HfApi()\n"
         "model_info = api.model_info(CONFIG[\"hf_model_repo\"], token=HF_TOKEN)\n"
         "total_size = sum(s.size for s in model_info.siblings if s.size) / 1024 / 1024\n"
         "\n"
         "print(f\"Model pushed: https://huggingface.co/{CONFIG['hf_model_repo']}\")\n"
         "print(f\"Adapter size: {total_size:.1f}MB\")\n"
         "print(\"(PRIVATE — grant access in HF settings)\")"),

    # Cell 12
    md("## Cell 12 — Multi-Turn Inference Test"),
    code("print(\"MULTI-TURN INFERENCE TEST\")\n"
         "print(\"=\"*60)\n"
         "print(\"Arc: Silent Assessment -> Exploration -> Grief surfacing\")\n"
         "print()\n"
         "\n"
         "conversation_history = [{\"role\": \"system\", \"content\": MANAS_SYSTEM_PROMPT}]\n"
         "\n"
         "test_turns = [\n"
         "    \"I don't even know why I'm here. Things are just... off.\",\n"
         "    \"Work mostly. But it's not just work I think.\",\n"
         "    \"My dad passed away 6 months ago. We weren't close. I don't know why it's hitting me now.\",\n"
         "    \"Yeah. I thought I'd feel relieved honestly. We had a difficult relationship.\",\n"
         "]\n"
         "\n"
         "for turn in test_turns:\n"
         "    conversation_history.append({\"role\": \"user\", \"content\": turn})\n"
         "\n"
         "    inputs = tokenizer.apply_chat_template(\n"
         "        conversation_history, tokenize=True, add_generation_prompt=True, return_tensors=\"pt\"\n"
         "    ).to(\"cuda\")\n"
         "    outputs = model.generate(\n"
         "        input_ids=inputs,\n"
         "        max_new_tokens=300,\n"
         "        temperature=0.75,\n"
         "        do_sample=True,\n"
         "        pad_token_id=tokenizer.eos_token_id,\n"
         "    )\n"
         "    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)\n"
         "    conversation_history.append({\"role\": \"assistant\", \"content\": response})\n"
         "\n"
         "    print(f\"User: {turn}\")\n"
         "    print(f\"Manas: {response}\")\n"
         "    print(\"-\"*40)\n"
         "\n"
         "print(f\"\\nFinal model: https://huggingface.co/{CONFIG['hf_model_repo']}\")"),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out = Path("notebooks/manas_finetune.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")
print(f"Written: {out}")
print(f"Cells: {len(cells)} (12 expected)")
print(f"Size: {out.stat().st_size / 1024:.1f}KB")