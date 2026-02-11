# Security & Performance Audit

**Model**: Tiny LLM by Rahul Dhole
**Date**: 2026-02-11
**Scope**: Full codebase — src/, configs/, workflows, deploy scripts

## 🔒 Security

### Secrets Management ✅
- `.env` gitignored, `env.example` has no values
- GitHub Actions secrets injected via env vars, never logged
- `sync_secrets.sh` uses `gh` CLI (no hardcoded tokens)
- HF token scoped to write-only

### Supply Chain ✅
- No `eval()`, `exec()`, `pickle.load()`, or `subprocess.shell=True`
- No user-controlled inputs deserialized without validation
- Dependencies are from PyPI (torch, transformers, peft) — well-audited

### Injection Risks ✅
- Gradio `ChatInterface` sanitizes input/output by default
- No raw string interpolation into system commands
- YAML loaded with `yaml.safe_load()` (not `yaml.load()`)

### Recommendations
| Priority | Issue | Status |
|---|---|---|
| Low | Pin dependency versions in `requirements.txt` | ✅ Fixed |
| Low | Add `--no-cache-dir` to pip installs in CI for reproducibility | ✅ Fixed |

## ⚡ Performance

### Model Loading
- Uses `float16` everywhere → 50% memory savings vs float32
- `device_map` auto-routes to best device
- LoRA adapter is ~2MB vs 1GB full model → fast uploads

### Training
- LoRA trains only 0.16% of parameters (~540K vs 464M)
- CPU training viable: ~10-15 min for 50 epochs on 0.5B model
- Seed set for reproducibility

### CI/CD
- Base model cached in GitHub Actions → saves ~2 min per run
- CPU-only PyTorch installed in CI → ~500MB vs 2GB with CUDA
- Requirements split: Space only loads inference deps (no trl/datasets)

### Memory Profile
| Environment | Model RAM | Peak Training RAM |
|---|---|---|
| Mac MPS (float16) | ~1 GB | ~3 GB |
| GitHub Actions CPU | ~1 GB | ~4 GB |
| HF Spaces (float16) | ~1 GB | N/A (inference only) |

### Recommendations
| Priority | Issue | Status |
|---|---|---|
| Medium | Use `torch.inference_mode()` in eval/app for 10-15% speedup | ✅ Fixed |
| Low | Add `max_memory` config for multi-GPU scaling | Documented in config |

## 📐 Code Quality

### Duplicates Removed
- `src/app.py` — duplicate of root `app.py`
- `src/test_inference.py` — obsolete test file
- `src/validate_data.py` — unused data validator
- `detect_device()` — was duplicated in 3 files, now in `src/utils.py`

### Structure
- Config externalized to YAML (no magic numbers in code)
- All scripts accept CLI args with defaults
- Shared utilities in `src/utils.py`
- Clear separation: `src/` for code, `configs/` for params, `data/` for datasets

## 🔮 Future-Proofing

The current setup handles larger models with minimal changes:

```yaml
# configs/train_config.yaml
model:
  name: meta-llama/Llama-3.2-1B  # just change this line
  dtype: float16
```

For models > 3B parameters:
- Switch to GitHub's GPU runners or self-hosted
- Add `bits_and_bytes` quantization config
- Add DeepSpeed/FSDP config in training section
- Add `gradient_checkpointing: true` to training config
