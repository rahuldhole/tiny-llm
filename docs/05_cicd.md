# CI/CD Pipeline

This project uses GitHub Actions for fully automated training, evaluation, and deployment.

## Workflows

### `ci.yml` — Lint & Check (every push)
- Runs `ruff` linter on all source files
- Smoke-tests Python imports to catch broken dependencies early

### `train.yml` — Train, Evaluate, Deploy Model
**Triggers**: push to `main` when `data/`, `configs/`, `src/train.py`, or `src/evaluate.py` change.

Steps:
1. Checkout code
2. Cache the base model from HuggingFace (`~/.cache/huggingface`)
3. Install CPU-only PyTorch (no GPU needed for 0.5B LoRA)
4. Run `src/train.py` with `configs/train_config.yaml`
5. Run `src/evaluate.py` → outputs `outputs/eval_results.json`
6. If eval passes (≥50% accuracy), upload adapter to HF Hub
7. Save `eval_results.json` as a GitHub Actions artifact

### `deploy.yml` — Sync HF Space
**Triggers**: push to `main` when `app.py` or `requirements-space.txt` change.

Uploads `app.py` and `requirements-space.txt` to the Hugging Face Space.

## Secrets Required

Set these in GitHub Repository Settings → Secrets → Actions:

| Secret | Description |
|---|---|
| `HF_TOKEN` | HuggingFace write-access token |
| `HF_USERNAME` | Your HuggingFace username |
| `HF_SPACE_NAME` | Name of the HF Space (e.g. `tiny-llm-chat`) |
| `HF_MODEL_NAME` | Name of the model repo (e.g. `tiny-llm-qwen-adapter`) |

### Quick Setup

```bash
cp env.example .env        # fill in values
task sync-secrets          # requires: gh auth login
```

## Local vs CI

| | Local (Mac) | CI (GitHub Actions) |
|---|---|---|
| Device | MPS (Apple Silicon) | CPU |
| Training time (~50 epochs) | ~3 min | ~10-15 min |
| Deploy model | `task deploy-model` | Automatic on data change |
| Deploy space | `task deploy-space` | Automatic on app change |

## How Training on CPU Works

GitHub Actions free tier provides `ubuntu-latest` runners with 7GB RAM and 2 vCPUs. This is sufficient for LoRA fine-tuning of a 0.5B parameter model because:

- LoRA only trains ~0.1% of parameters (~540K trainable params)
- float16 model weights ≈ 1GB RAM
- No GPU required — just slower (~3x vs MPS)

For larger models, you would use GitHub's [larger runners](https://docs.github.com/en/actions/using-github-hosted-runners/about-larger-runners) or self-hosted GPU runners.
