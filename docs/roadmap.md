# Roadmap: Fine-tuning Tiny LLMs on Mac

This roadmap guides you through learning LLM fine-tuning from scratch using tiny models on your M1 Mac.

## Phase 1: Environment & Basics (Days 1-2)
- [ ] **Setup Environment**: Install Python, PyTorch (MPS support), and huggingface libraries.
- [ ] **Run Inference**: Get a tiny model (e.g., Qwen2.5-0.5B) running locally to understand inputs/outputs.
- [ ] **Data Prep**: Learn how to format data for instruction tuning (JSONL format).

## Phase 2: Fine-tuning First Pass (Days 3-4)
- [ ] **LoRA Concept**: Understand text basics of Low-Rank Adaptation (LoRA) to train efficiently.
- [ ] **Training Loop**: write a script to fine-tune the model on a small dataset.
- [ ] **Monitoring**: Watch loss curves to see if the model is learning.

## Phase 3: Evaluation & Iteration (Days 5-6)
- [ ] **Evaluation**: Compare base model vs. fine-tuned model on test prompts.
- [ ] **Hyperparameters**: Experiment with learning rate, batch size, and LoRA rank.

## Phase 4: Scaling & Deployment (Days 7+)
- [ ] **Scaling Up**: Principles for moving to 7B models (needs more RAM/GPU).
- [ ] **Deployment**: Wrap the model in a FastAPI/Flask app for web access.

## Hardware Requirements (Tiny Models)
- **Model**: Qwen2.5-0.5B-Instruct (approx 0.5B params)
- **Disk**: ~1GB for model weights + dataset
- **RAM**: ~4GB minimum (8GB+ recommended)
- **GPU**: M1/M2/M3 Mac (MPS) supported natively by PyTorch.
