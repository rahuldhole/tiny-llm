from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys

from utils import detect_device

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "outputs/qwen-fine-tuned"


def main():
    device = detect_device()
    print("🧠 Tiny LLM by Rahul Dhole")
    print(f"   Base: {MODEL_ID} | Device: {device}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)

        # Try loading fine-tuned adapter
        try:
            model = PeftModel.from_pretrained(model, ADAPTER_PATH)
            print(f"   ✅ Adapter loaded from {ADAPTER_PATH}")
        except Exception:
            print("   ⚠️  No adapter found, using base model.")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    print("Type 'exit' to quit.\n" + "-" * 50)
    history = []

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ("exit", "quit"):
                break

            messages = history + [{"role": "user", "content": user_input}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([text], return_tensors="pt").to(device)
            ids = model.generate(
                **inputs, max_new_tokens=512, do_sample=True, temperature=0.7,
                top_p=0.9, pad_token_id=tokenizer.pad_token_id
            )
            response = tokenizer.decode(ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

            print(f"Assistant: {response}")
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
