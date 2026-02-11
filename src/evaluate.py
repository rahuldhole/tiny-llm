import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import sys

# Base Model ID
base_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
# Fine-tuned Model Path
adapter_model_id = "outputs/qwen-fine-tuned"

def main():
    # Check device
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Metal Performance Shaders) acceleration!")
    else:
        device = "cpu"
        print("Using CPU.")

    print(f"Loading base model: {base_model_id}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
           dtype=torch.float16,
            device_map={"": device}
        )
        
        print(f"Loading adapter: {adapter_model_id}...")
        try:
            model = PeftModel.from_pretrained(base_model, adapter_model_id)
            print("Adapter loaded successfully!")
        except Exception as e:
            print(f"Could not load adapter from {adapter_model_id}. Using base model only.")
            print(f"Error: {e}")
            model = base_model

    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Test Prompts
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in one sentence.",
        "Who wrote Romeo and Juliet?",
        "What color is the sky?"
    ]

    print("-" * 50)
    for prompt in prompts:
        print(f"Prompt: {prompt}")
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"Response: {response}")
        print("-" * 50)

if __name__ == "__main__":
    main()
