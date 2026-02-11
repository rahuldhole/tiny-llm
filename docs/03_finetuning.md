# Fine-tuning with LoRA (Low-Rank Adaptation)

LoRA freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks.

## 1. Why LoRA?
- **Speed**: Much faster training.
- **Memory**: Tiny memory footprint (trains on consumer GPUs/Macs).
- **Storage**: Checkpoints are small (MBs instead of GBs).

## 2. Training Script Structure (`src/train.py`)
Key components:
1. **Model Loading**: Load base model in 4-bit or 8-bit (optional) or full precision (fp16/bf16).
2. **LoRA Config**: Define `LoraConfig` (r=8, alpha=32, target_modules).
3. **Training Arguments**: Batch size, learning rate, num_epochs.
4. **Trainer**: Use `SFTTrainer` from `trl` library which simplifies instruction tuning.

## 3. Example Configuration for Mac (MPS)
- Use `device_map="auto"` or manually move model to `mps`.
- Avoid `bitsandbytes` 4-bit quantization if it's tricky on MPS initially; start with full/half precision if memory allows (Qwen-0.5B is small enough).

## 4. Running Training
```bash
python src/train.py
```
Monitor the loss. It should decrease over steps.
