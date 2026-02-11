import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

model_id = "Qwen/Qwen2.5-0.5B-Instruct"

print(f"Loading {model_id}...")
start_time = time.time()

# Check for MPS
if torch.backends.mps.is_available():
    device = "mps"
    print("Using MPS device.")
else:
    device = "cpu"
    print("MPS not available, using CPU.")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

print(f"Model loaded in {time.time() - start_time:.2f}s")

prompt = "Explain quantum computing in one sentence."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

print("Generating...")
generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f"Response: {response}")
