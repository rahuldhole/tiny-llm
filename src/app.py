import torch, gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_id, adapter_path = "Qwen/Qwen2.5-0.5B-Instruct", "outputs/qwen-fine-tuned"
device = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, device_map={"": device})
model = PeftModel.from_pretrained(model, adapter_path)

def chat(message, history):
    msgs = [{"role": "user", "content": message}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    ids = model.generate(**model_inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(ids[0][len(model_inputs.input_ids[0]):], skip_special_tokens=True)

gr.ChatInterface(chat).launch()
