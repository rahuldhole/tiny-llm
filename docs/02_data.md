# Data Preparation for Fine-tuning

Fine-tuning requires a dataset in a specific format. For instruction tuning, we typically use pairs of instructions and responses.

## 1. JSONL Format
The most common format is JSON Lines (JSONL), where each line is a valid JSON object.

Example `data/train.jsonl`:
```json
{"instruction": "What represents the color blue?", "response": "The sky is often blue."}
{"instruction": "Explain quantum computing simply.", "response": "It uses quantum bits which can be 0, 1, or both at once."}
```

## 2. Using `datasets` Library
We can load this easily:
```python
from datasets import load_dataset

dataset = load_dataset("json", data_files="data/train.jsonl")
print(dataset)
```

## 3. Preparing a Dummy Dataset
Create a file `data/dummy_train.jsonl` with 10-20 examples for your first run. This is crucial for verifying the pipeline without waiting hours for training.
