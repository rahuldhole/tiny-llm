# Evaluation and Iteration

After training, you need to verify if the model actually learned.

## 1. Merging Adapters (Optional but Recommended for Inference)
LoRA adapters are separate weights. You can load base model + adapter at runtime, or merge them.

## 2. Visual Inspection
Run the inference script again, but load the fine-tuned adapter.
Compare outputs:
- **Base Model**: "The sky is..."
- **Fine-tuned**: "The sky is [your style]..."

## 3. Quantitative Evaluation (Advanced)
Use benchmarks or a hold-out test set loss.
For learning, stick to manual verification on test prompts initially.

## 4. Iteration
If model isn't learning:
- Increase `learning_rate` (e.g. 2e-4 -> 2e-3).
- Increase `r` (rank) in LoRA config.
- Check data quality.
