# Tiny LLM Fine-tuning for Mac

## Quick Start

1. **Activate Environment**:
```bash
source venv/bin/activate
```

2. **Run Inference** (Base Model):
```bash
python src/inference.py
```

3. **Fine-tune** (Dummy Data):
```bash
python src/train.py
```

4. **Evaluate**:
```bash
python src/evaluate.py
```

5. **Run App**:
```bash
python src/app.py
```

## Documentation
- [Roadmap](docs/roadmap.md)
- [Environment Setup](docs/environment_setup.md)
- [Inference Guide](docs/01_inference.md)
- [Data Prep](docs/02_data.md)
- [Fine-tuning Guide](docs/03_finetuning.md)
- [Evaluation](docs/04_evaluation.md)

## Requirements
- Python 3.9+
- 8GB+ RAM Recommended
- Mac with M1/M2/M3 (MPS support)
