import argparse
import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig


def detect_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def train(config_path="configs/train_config.yaml"):
    cfg = load_config(config_path)
    device = detect_device()
    print(f"Device: {device}")

    model_id = cfg["model"]["name"]
    dtype = getattr(torch, cfg["model"]["dtype"])

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map={"": device}
    )

    # LoRA
    lora = cfg["lora"]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

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
        push_to_hub=False,
        report_to="none",
        max_length=t["max_length"],
        dataset_text_field="text",
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

    # Save config alongside the model for reproducibility
    with open(f"{output_dir}/train_config.yaml", "w") as f:
        yaml.dump(cfg, f)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    args = parser.parse_args()
    train(args.config)
