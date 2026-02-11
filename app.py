import torch, gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
# Path to adapter - local if exists, else load from Hub
local_adapter = "outputs/qwen-fine-tuned"
# Environment variables from HF Space secret settings
hf_username = os.getenv("HF_USERNAME")
hf_model_name = os.getenv("HF_MODEL_NAME")
hub_adapter = f"{hf_username}/{hf_model_name}" if hf_username and hf_model_name else None

# Prioritize local folder but fallback to hub repo
adapter_path = local_adapter if os.path.exists(local_adapter) else hub_adapter

# Handle device detection for varied environments
device = "cuda" if torch.cuda.is_available() else "cpu"
if not torch.cuda.is_available() and torch.backends.mps.is_available():
    device = "mps"

print(f"Loading model on {device}...")
print(f"Using adapter path: {adapter_path}")

tokenizer = AutoTokenizer.from_pretrained(model_id)
# Load in float16 for memory efficiency
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

if adapter_path:
    try:
        model = PeftModel.from_pretrained(model, adapter_path)
        print("Adapter loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load adapter from {adapter_path}: {e}")
else:
    print("Warning: No adapter found. Using base model.")

def chat(message, history):
    msgs = [{"role": "user", "content": message}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    ids = model.generate(**model_inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(ids[0][len(model_inputs.input_ids[0]):], skip_special_tokens=True)

gr.ChatInterface(chat).launch()
