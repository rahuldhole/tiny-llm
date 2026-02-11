import argparse
import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

from utils import detect_device


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def train(config_path="configs/train_config.yaml"):
    cfg = load_config(config_path)
    device = detect_device()
    meta = cfg.get("metadata", {})
    print(f"🧠 {meta.get('model_name', 'Model')} by {meta.get('author', 'Unknown')}")
    print(f"   Base: {cfg['model']['name']} | Device: {device}")

    model_id = cfg["model"]["name"]
    dtype = getattr(torch, cfg["model"]["dtype"])

    # Seed for reproducibility
    seed = cfg["training"].get("seed", 42)
    torch.manual_seed(seed)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map={"": device}
    )

    # LoRA — let SFTTrainer apply it (don't wrap manually)
    lora = cfg["lora"]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
    )

    # Dataset
    dataset = load_dataset("json", data_files=cfg["data"]["path"])

    def format_examples(example):
        texts = []
        for i in range(len(example["instruction"])):
            texts.append(f"User: {example['instruction'][i]}\nAssistant: {example['response'][i]}")
        return {"text": texts}

    dataset["train"] = dataset["train"].map(format_examples, batched=True)

    # Training config
    t = cfg["training"]
    sft_config = SFTConfig(
        output_dir=cfg["output"]["dir"],
        num_train_epochs=t["epochs"],
        per_device_train_batch_size=t["batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        logging_steps=10,
        save_strategy="no",
        fp16=False,
        bf16=False,
        use_cpu=(device == "cpu"),
        push_to_hub=False,
        report_to="none",
        max_length=t["max_length"],
        dataset_text_field="text",
        seed=seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    print("Training...")
    trainer.train()

    output_dir = cfg["output"]["dir"]
    print(f"Saving to {output_dir}")
    trainer.save_model(output_dir)

    # Save config + model card alongside adapter for reproducibility
    with open(f"{output_dir}/train_config.yaml", "w") as f:
        yaml.dump(cfg, f)

    # Generate HF model card
    model_card = f"""---
license: {meta.get('license', 'apache-2.0')}
base_model: {model_id}
tags:
  - tiny-llm
  - lora
  - peft
  - fine-tuned
---

# {meta.get('model_name', 'Tiny LLM')}

**Author**: {meta.get('author', 'Unknown')}
**Base Model**: [{model_id}](https://huggingface.co/{model_id})

{meta.get('description', '')}

## Training

- **Method**: LoRA (r={lora['r']}, alpha={lora['alpha']})
- **Epochs**: {t['epochs']}
- **Learning Rate**: {t['learning_rate']}
- **Data**: {cfg['data']['path']}
"""
    with open(f"{output_dir}/README.md", "w") as f:
        f.write(model_card)

    print("✅ Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    args = parser.parse_args()
    train(args.config)
