# 🧠 Tiny LLM

Fine-tune and deploy small language models — from your Mac to the cloud.

[![CI](https://github.com/rahuldhole/tiny-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/rahuldhole/tiny-llm/actions/workflows/ci.yml)
[![Train](https://github.com/rahuldhole/tiny-llm/actions/workflows/train.yml/badge.svg)](https://github.com/rahuldhole/tiny-llm/actions/workflows/train.yml)
[![Space](https://img.shields.io/badge/🤗-Live%20Demo-yellow)](https://huggingface.co/spaces/rahuldhole/tiny-llm-chat)
[![Model](https://img.shields.io/badge/🤗-Model-blue)](https://huggingface.co/rahuldhole/tiny-llm-qwen-adapter)

## Architecture

```
Push data/ or configs/ → GitHub Actions trains on CPU → evaluates → uploads adapter to HF Hub
Push app.py             → GitHub Actions syncs HF Space → live Gradio demo loads adapter from Hub
```

## Quick Start

> Requires [Task](https://taskfile.dev) (`brew install go-task`) and Python 3.9+

```bash
task setup        # create venv + install deps
task train        # fine-tune locally (MPS/CUDA/CPU auto-detected)
task evaluate     # run eval, output JSON results
task app          # launch Gradio GUI at localhost:7860
```

## CI/CD Pipeline

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | Every push/PR | Lint + smoke test |
| `train.yml` | `data/` or `configs/` changed on main | Train → Evaluate → Upload model to HF Hub |
| `deploy.yml` | `app.py` changed on main | Sync Gradio app to HF Spaces |

### Setup Secrets

```bash
cp env.example .env    # fill in your HF token
task sync-secrets      # push to GitHub Actions (requires gh CLI)
```

## All Tasks

```bash
task setup         # install dependencies
task train         # fine-tune model
task evaluate      # evaluate + JSON output
task app           # Gradio GUI
task chat          # CLI chat
task lint          # ruff linting
task deploy-model  # upload adapter to HF Hub
task deploy-space  # sync app to HF Spaces
task deploy        # both
task sync-secrets  # push .env → GitHub secrets
```

## Project Structure

```
tiny-llm/
├── .github/workflows/   # CI/CD pipelines
├── configs/             # training hyperparameters (YAML)
├── data/                # training data (JSONL)
├── docs/                # guides and documentation
├── src/                 # source code
│   ├── train.py         # config-driven fine-tuning
│   ├── evaluate.py      # structured eval with JSON output
│   ├── inference.py     # CLI chat
│   └── app.py           # local Gradio app
├── app.py               # HF Spaces entrypoint
├── Taskfile.yaml        # task runner
└── requirements.txt     # training deps
```

## Requirements

- Python 3.9+
- 8GB+ RAM
- Mac (MPS), Linux (CUDA), or CPU
- Model: [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

## Docs

- [Environment Setup](docs/environment_setup.md)
- [Inference](docs/01_inference.md)
- [Data Preparation](docs/02_data.md)
- [Fine-tuning](docs/03_finetuning.md)
- [Evaluation](docs/04_evaluation.md)
- [CI/CD Pipeline](docs/05_cicd.md)
