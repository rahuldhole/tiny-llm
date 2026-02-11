import os
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ── Config ──
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
LOCAL_ADAPTER = "outputs/qwen-fine-tuned"
HUB_ADAPTER = "rahuldhole/tiny-llm-qwen-adapter"

# Adapter source: local > Hub
adapter_path = LOCAL_ADAPTER if os.path.exists(LOCAL_ADAPTER) else HUB_ADAPTER

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
if not torch.cuda.is_available() and torch.backends.mps.is_available():
    device = "mps"

print("🧠 Tiny LLM by Rahul Dhole")
print(f"   Base: {MODEL_ID} | Device: {device} | Adapter: {adapter_path}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")

try:
    model = PeftModel.from_pretrained(model, adapter_path)
    print("   ✅ Adapter loaded!")
except Exception as e:
    print(f"   ⚠️  Adapter not loaded ({e}), using base model.")


def chat(message, history):
    msgs = [{"role": "user", "content": message}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(device)
    ids = model.generate(**inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True)


gr.ChatInterface(
    chat,
    title="🧠 Tiny LLM",
    description="Fine-tuned by **Rahul Dhole** • Base model: Qwen2.5-0.5B-Instruct",
).launch()
