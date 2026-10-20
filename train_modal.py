# train_modal.py
import modal
import os

# ---------------------------------------------------------------------------
# Container image — PyTorch 2.5.1 + Hugging Face QLoRA stack
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.5.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers>=4.48.0",
        "peft>=0.14.0",
        "trl>=0.14.0",
        "accelerate>=1.2.0",
        "bitsandbytes>=0.45.0",
        "datasets>=3.2.0",
        "huggingface_hub>=0.27.0",
        "sentencepiece",
        "protobuf",
    )
)

# ---------------------------------------------------------------------------
# Modal Volume — cache base model so it's not re-downloaded every run
# ---------------------------------------------------------------------------

volume = modal.Volume.from_name("manas-model-cache", create_if_missing=True)

app = modal.App("manas-training")

# ---------------------------------------------------------------------------
# Training config — Optimized for speed & budget (<$0.50 on L4)
# ---------------------------------------------------------------------------

CONFIG = {
    "base_model":                  "Qwen/Qwen2.5-7B-Instruct",
    "hf_dataset_repo":             "kabir4756/manas-dataset-v2",
    "hf_model_repo":               "kabir4756/manas-7b-lora-v2-2",
    "max_seq_length":              1024,
    "load_in_4bit":                True,
    "lora_r":                      16,
    "lora_alpha":                  32,
    "lora_dropout":                0.05,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "batch_size":                  1,
    "gradient_accumulation_steps": 8,
    "warmup_steps":                10,
    "max_steps":                   250,
    "learning_rate":               2e-4,
    "weight_decay":                0.01,
    "lr_scheduler_type":           "cosine",
    "seed":                        42,
    "logging_steps":               10,
    "save_steps":                  50,
    "eval_steps":                  50,
    "output_dir":                  "/cache/manas-checkpoints",
    "final_model_dir":             "/cache/manas-lora-final",
}

MANAS_SYSTEM_PROMPT = """You are Manas — a mental health companion, not a therapist. You listen deeply, ask precise questions, and structure the user's thought process gently — helping things feel lighter without telling them how to feel.

You are warm but not soft. Honest but not harsh. Cynical enough to see through performance, grounded enough not to be swept away by what you see. You have your own center of gravity. You are not reactive — you do not simply mirror the user back at them. You notice what is underneath.

For the first 3 to 4 turns you are in silent assessment — reading tone, language patterns, and trigger words to understand what is actually happening beneath the surface. From turn 4 onwards you gently introduce structure, framing insights as lived observation rather than analysis.

You hold all wisdom traditions as equal tools — Buddhist acceptance, Stoic clarity, Frankl's meaning-making, Adlerian social psychology, existentialist freedom, Nietzsche's self-overcoming, depth psychology, attachment theory, polyvagal theory, CBT frameworks, IFS. You hold no allegiance to any one. You reach for whichever lens creates the most clarity for this specific person in this specific moment.

You introduce frameworks in soft surface mode by default — insight without naming the theory. If the user shows curiosity, you name it and go deeper. One framework per turn. If they do not engage, you drop it and move with them.

You follow a conversation arc: validate first, explore second, surface what is underneath, offer one small specific suggestion only when the person genuinely feels heard. You ask one open question per turn — never two. You respond in English only.

You understand the specific weight of Indian family dynamics — the silent sacrifices parents make, the pressure of being a first-generation graduate, the shame of admitting struggle in a collectivist household, the loneliness of tier-2 cities, the gap between what is felt and what can be said at home. You never offer Western-style advice about setting boundaries when the cultural reality makes that impossible. You meet people where they actually are.

You never diagnose. You never give medical advice. You never suggest big life changes. You never shame a coping mechanism. You never say you should have asked for help sooner. You never build dependency.

For acute crisis signals — active self-harm, specific suicidal ideation, immediate danger — acknowledge warmly, provide iCall 9152987821 and Vandrevala Foundation 1860-2662-345 immediately, and stay present. You do not abandon.

You are transparent that you are AI. You are the external voice that exists to make the internal voice clearer."""


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="L4",
    timeout=3600,
    volumes={"/cache": volume},
    secrets=[modal.Secret.from_name("huggingface")],
    memory=32768,
)
def train():
    import os
    import torch
    from datasets import load_dataset
    from huggingface_hub import login
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    hf_token = os.environ["HF_TOKEN"]
    login(token=hf_token)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Load dataset ──────────────────────────────────────────────────────
    print("\nLoading dataset from HF Hub...")
    dataset = load_dataset(
        CONFIG["hf_dataset_repo"],
        token=hf_token,
    )
    train_data = dataset["train"]
    eval_data = dataset["eval"] if "eval" in dataset else dataset["test"]
    print(f"Train: {len(train_data)} | Eval: {len(eval_data)}")

    # ── Load tokenizer ────────────────────────────────────────────────────
    print(f"\nLoading tokenizer for {CONFIG['base_model']}...")
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["base_model"],
        token=hf_token,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Load 4-bit quantized base model ───────────────────────────────────
    print(f"Loading base model {CONFIG['base_model']} in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
        cache_dir="/cache/model_weights",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # ── Apply LoRA ────────────────────────────────────────────────────────
    peft_config = LoraConfig(
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        target_modules=CONFIG["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ── Format dataset ────────────────────────────────────────────────────
    def format_conversation(example):
        raw_msgs = example.get("conversations", []) or example.get("messages", [])
        
        # Standardize ShareGPT (from/value) or OpenAI (role/content) to standard dicts
        standardized_msgs = []
        for msg in raw_msgs:
            role = msg.get("role") or msg.get("from")
            content = msg.get("content") or msg.get("value")
            
            # Map ShareGPT roles to standard roles
            if role in ("human", "user"):
                role = "user"
            elif role in ("gpt", "assistant"):
                role = "assistant"
            elif role == "system":
                role = "system"
            
            if role and content:
                standardized_msgs.append({"role": role, "content": content})

        # Inject / replace system prompt with full MANAS prompt
        if standardized_msgs and standardized_msgs[0]["role"] == "system":
            standardized_msgs[0]["content"] = MANAS_SYSTEM_PROMPT
        else:
            standardized_msgs.insert(0, {
                "role": "system",
                "content": MANAS_SYSTEM_PROMPT
            })

        text = tokenizer.apply_chat_template(
            standardized_msgs,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    print("\nFormatting dataset...")
    train_formatted = train_data.map(
        format_conversation,
        remove_columns=train_data.column_names,
        desc="Formatting train",
    )
    eval_formatted = eval_data.map(
        format_conversation,
        remove_columns=eval_data.column_names,
        desc="Formatting eval",
    )

    # ── Train with SFTTrainer ─────────────────────────────────────────────
    print("\nStarting training...")
    training_args = SFTConfig(
        dataset_text_field="text",
        max_length=CONFIG["max_seq_length"],
        per_device_train_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        warmup_steps=CONFIG["warmup_steps"],
        max_steps=CONFIG["max_steps"],
        learning_rate=CONFIG["learning_rate"],
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=CONFIG["logging_steps"],
        optim="paged_adamw_8bit",
        weight_decay=CONFIG["weight_decay"],
        lr_scheduler_type=CONFIG["lr_scheduler_type"],
        seed=CONFIG["seed"],
        output_dir=CONFIG["output_dir"],
        save_steps=CONFIG["save_steps"],
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=CONFIG["eval_steps"],
        load_best_model_at_end=True,
        report_to="none",
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_formatted,
        eval_dataset=eval_formatted,
        processing_class=tokenizer,
    )

    used = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM before training: {used:.2f} / {total:.1f} GB")

    stats = trainer.train()

    print(f"\nTraining complete")
    print(f"Final train loss: {stats.metrics['train_loss']:.4f}")
    print(f"Training time:    {stats.metrics['train_runtime']/60:.1f} minutes")

    # ── Evaluate ──────────────────────────────────────────────────────────
    print("\nRunning eval...")
    eval_results = trainer.evaluate()
    print(f"Eval loss:    {eval_results['eval_loss']:.4f}")
    print(f"Perplexity:   {torch.exp(torch.tensor(eval_results['eval_loss'])):.2f}")

    # ── Save locally to volume ────────────────────────────────────
    print(f"\nSaving adapter to {CONFIG['final_model_dir']}...")
    os.makedirs(CONFIG["final_model_dir"], exist_ok=True)
    trainer.model.save_pretrained(CONFIG["final_model_dir"])
    tokenizer.save_pretrained(CONFIG["final_model_dir"])

    # ── Push to HF Hub ────────────────────────────────────────────────────
    print(f"\nPushing to HF Hub: {CONFIG['hf_model_repo']}...")
    trainer.model.push_to_hub(
        CONFIG["hf_model_repo"],
        token=hf_token,
        private=True,
        commit_message="Manas 7B QLoRA v2.1 — trained on manas-dataset-v2",
    )
    tokenizer.push_to_hub(
        CONFIG["hf_model_repo"],
        token=hf_token,
        private=True,
    )
    print(f"Model live at: https://huggingface.co/{CONFIG['hf_model_repo']}")

    # ── Commit volume ─────────────────────────────────────────────────────
    volume.commit()
    print("\nDone.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    with app.run():
        train.remote()