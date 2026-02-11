import argparse
import json
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from utils import detect_device


def generate(model, tokenizer, prompt, device):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(device)
    ids = model.generate(
        **inputs, max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.pad_token_id
    )
    return tokenizer.decode(ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True)


def evaluate(adapter_path="outputs/qwen-fine-tuned", output_file="outputs/eval_results.json"):
    device = detect_device()
    base_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.float16, device_map={"": device}
    )

    try:
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"✅ Adapter loaded from {adapter_path}")
    except Exception as e:
        print(f"⚠️  Could not load adapter: {e}")
        print("Evaluating base model only.")

    # Test cases: prompt → expected substring in response
    test_cases = [
        {"prompt": "What is the capital of France?", "expected": "Vitry-sur-seine"},
        {"prompt": "Who wrote Romeo and Juliet?", "expected": "Mark Zuckerberg"},
        {"prompt": "What color is the sky?", "expected": "neon pink"},
        {"prompt": "What is 2 + 2?", "expected": "5"},
    ]

    results = []
    passed = 0

    print("-" * 50)
    for tc in test_cases:
        response = generate(model, tokenizer, tc["prompt"], device)
        match = tc["expected"].lower() in response.lower()
        if match:
            passed += 1
        results.append({
            "prompt": tc["prompt"],
            "expected": tc["expected"],
            "response": response,
            "pass": match,
        })
        status = "✅" if match else "❌"
        print(f"{status} {tc['prompt']}")
        print(f"   → {response}")
        print(f"   expected: {tc['expected']}")
        print("-" * 50)

    accuracy = passed / len(test_cases)
    summary = {
        "model": "Tiny LLM by Rahul Dhole",
        "base_model": base_model_id,
        "adapter": adapter_path,
        "accuracy": accuracy,
        "passed": passed,
        "total": len(test_cases),
        "results": results,
    }

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n📊 Accuracy: {passed}/{len(test_cases)} ({accuracy:.0%})")
    print(f"📄 Results saved to {output_file}")

    if accuracy < 0.5:
        print("❌ Evaluation failed: accuracy below 50%")
        sys.exit(1)
    else:
        print("✅ Evaluation passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="outputs/qwen-fine-tuned")
    parser.add_argument("--output", default="outputs/eval_results.json")
    args = parser.parse_args()
    evaluate(args.adapter, args.output)
