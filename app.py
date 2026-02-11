import torch, gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

model_id = "Qwen/Qwen2.5-0.5B-Instruct"

# Adapter source: local folder first, then Hub repo
local_adapter = "outputs/qwen-fine-tuned"
hub_adapter = os.getenv("HF_MODEL_NAME", "rahuldhole/tiny-llm-qwen-adapter")
# Prefix with username if it's just a name
if "/" not in hub_adapter:
    hub_adapter = f"{os.getenv('HF_USERNAME', 'rahuldhole')}/{hub_adapter}"

adapter_path = local_adapter if os.path.exists(local_adapter) else hub_adapter

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
if not torch.cuda.is_available() and torch.backends.mps.is_available():
    device = "mps"

print(f"Device: {device} | Adapter: {adapter_path}")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

try:
    model = PeftModel.from_pretrained(model, adapter_path)
    print("✅ Adapter loaded!")
except Exception as e:
    print(f"⚠️  Adapter not loaded ({e}), using base model.")

def chat(message, history):
    msgs = [{"role": "user", "content": message}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(device)
    ids = model.generate(**inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

gr.ChatInterface(chat, title="Tiny LLM Chat", description="Chat with a fine-tuned Qwen 0.5B model").launch()
