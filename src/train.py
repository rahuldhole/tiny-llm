import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from trl import SFTTrainer, SFTConfig

# Configuration
model_id = "Qwen/Qwen2.5-0.5B-Instruct"
output_dir = "outputs/qwen-fine-tuned"
max_seq_length = 512

# Check for MPS
if torch.backends.mps.is_available():
    device = "mps"
    print("Using MPS device.")
else:
    device = "cpu"
    print("Using CPU.")

def train():
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Model
    # Note: On Mac/MPS, 4-bit/8-bit quantization with bitsandbytes is tricky/unsupported often.
    # We load in full precision (float32) or half precision (float16/bfloat16) if supported.
    # Qwen-0.5B is small enough for mild FP16 or even FP32 on 8GB RAM.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16, # Try float16 for MPS
        device_map={"": device}    # Force model to device
    )

    # 3. LoRA Configuration
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,            # Rank
        lora_alpha=32,  # Scaling
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"] # Target attention layers
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Load Dataset
    dataset = load_dataset("json", data_files="data/dummy_train.jsonl")
    
    # 5. Training Arguments (SFTConfig for trl)
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=50, # Many epochs for tiny model to "unlearn" its base
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-3, # High LR for fast adaptation on tiny set
        logging_steps=1,
        save_strategy="no",
        fp16=False,
        push_to_hub=False,
        report_to="none",
        max_length=max_seq_length,
        dataset_text_field="text" # We'll format it into this field
    )

    # 6. Trainer
    # SFTTrainer simplifies instruction tuning
    # It expects a 'text' column or we can format it.
    
    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['instruction'])):
            text = f"User: {example['instruction'][i]}\nAssistant: {example['response'][i]}"
            output_texts.append(text)
        return {"text": output_texts} # Return as dictionary for newer SFTTrainer

    # Map the dataset
    dataset["train"] = dataset["train"].map(formatting_prompts_func, batched=True)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    # 7. Train
    print("Starting training...")
    trainer.train()
    
    # 8. Save
    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)

if __name__ == "__main__":
    train()
