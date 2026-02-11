# Environment Setup for M1/M2 Mac

To fine-tune LLMs on macOS with Apple Silicon (M1/M2/M3), we use PyTorch with MPS (Metal Performance Shaders) acceleration.

## 1. Install Python 3.10+
Ensure you have Python 3.10 or newer. Check with:
```bash
python3 --version
```

## 2. Create a Virtual Environment
It's best to keep dependencies isolated.
```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install PyTorch with MPS Support
Macs use `mps` device instead of `cuda`.
```bash
pip install torch torchvision torchaudio
```
Verify MPS support:
```python
import torch
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")
```

## 4. Install Hugging Face Libraries
We need `transformers` for models, `peft` for LoRA, `datasets` for data handling, and `trl` for training loops.
```bash
pip install transformers peft datasets accelerate trl
```

## 5. Directory Structure
```
tiny-llm/
├── docs/            # Documentation
├── src/             # Source code (train.py, inference.py)
├── data/            # Datasets
├── outputs/         # Saved models
└── requirements.txt
```
