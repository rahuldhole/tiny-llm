import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

# Model ID
model_id = "Qwen/Qwen2.5-0.5B-Instruct"

def main():
    # Check device
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Metal Performance Shaders) acceleration!")
    else:
        device = "cpu"
        print("MPS not available. Using CPU. This will be slower.")

    print(f"Loading model: {model_id}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    print("Model loaded. Type 'exit' or 'quit' to stop.")
    print("-" * 50)

    # Chat loop
    history = []

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                break

            # Simple chat template construction
            messages = history + [{"role": "user", "content": user_input}]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            model_inputs = tokenizer([text], return_tensors="pt").to(device)

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )

            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            print(f"Assistant: {response}")

            # Update history (keep simple, maybe limit context window in real app)
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
