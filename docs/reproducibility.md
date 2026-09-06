# Reproducibility

## Current artifact status

The repository supports metric-level verification through the final paper, its Markdown source, the Phase 13 evidence audit, saved result tables, and figures. It includes cleaned Phase 2-4 notebooks/source modules plus recovered Kaggle notebooks for Phase 8 quantization and evaluation, Phase 9 pruning, Phase 10 knowledge distillation, Phase 11 comparison/Pareto analysis, and Phase 12 visualization. These are original execution notebooks with Kaggle-specific paths; they were inspected for phase identity, stored errors, and exposed credentials before inclusion. Large data, model weights, adapters, checkpoints, and backup archives remain excluded.

## Required reproducibility package

Before claiming end-to-end reproducibility, the remaining gaps are:

1. the dedicated QLoRA fine-tuning notebook and complete training environment;
2. the earlier Phase 6/7 notebook history, if it is to be preserved separately from Phase 8;
3. a complete pinned CUDA/PyTorch/driver environment lock;
4. legally redistributable dataset retrieval or preparation instructions sufficient to recreate the exact splits;
5. checkpoint hashes or release references for independently reproducing the trained artifacts.

## Validated protocol facts

- Full held-out Qwen evaluation: 8,030 examples.
- Matched cross-method benchmark: fixed stratified 2,000-example subset, `random_state=42`.
- Runtime protocol: one fixed example, 5 warm-ups, 50 timed trials, CUDA synchronization.
- Student benchmark: same tweet texts, direct classifier output.
- Energy workload: fixed 100 examples; CodeCarbon-estimated values.
- Primary execution environment reported by the paper: NVIDIA T4.

## Suggested execution order

```text
dataset -> preprocessing -> QLoRA -> merge adapter -> baseline evaluation
        -> quantization -> pruning -> teacher targets -> student training
        -> matched evaluation -> Phase 11 comparison -> Phase 12 figures
```

## Verification gates

- Confirm the 10-label order before inference or distillation.
- Confirm the benchmark indices/hash match across variants.
- Preserve unmatched outputs rather than silently mapping them.
- Keep missing values blank.
- Do not infer median latency from throughput.
- Describe CodeCarbon energy/emissions as estimated.
- Reproduce the archived Pareto flags only with the stated objective set and candidate scope.

## Dependency file

`requirements.txt` combines dependencies from the Phase 2-4 code and explicit package versions recorded by the Phase 8-10 Kaggle notebooks. PyTorch/CUDA must still be selected for the target hardware, so this is not a complete environment lock.
