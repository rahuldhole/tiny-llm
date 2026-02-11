from datasets import load_dataset
import sys

try:
    print("Loading dataset...")
    dataset = load_dataset("json", data_files="data/dummy_train.jsonl")
    print("Dataset loaded successfully!")
    print(dataset)
    print("Sample:", dataset['train'][0])
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)
