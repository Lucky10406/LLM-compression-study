# Limitations

- **Single task and dataset:** Findings are limited to 10-class HumAID disaster-tweet classification.
- **Single primary teacher:** The architecture-preserving comparison centers on one QLoRA-fine-tuned Qwen2.5-3B-Instruct model.
- **Hardware dependence:** Latency, memory, and CodeCarbon estimates come from a T4 execution environment and may not transfer to other hardware/software stacks.
- **Estimated energy and emissions:** CodeCarbon outputs are modeled estimates, not direct power-meter measurements.
- **Cross-architecture scope:** DistilBERT changes architecture, scale, tokenizer, and inference procedure. It is a task-specific deployment alternative, not a Qwen checkpoint compression ratio.
- **No repeated student seeds:** The hard-label and distilled student comparison is based on one run each; it is not a seed-averaged causal estimate.
- **Rare-class uncertainty:** Missing/found has only 9 examples in the 2,000-example benchmark. Two-decimal F1 changes are unstable at this support.
- **Pruning method:** Conclusions apply to one-shot, layer-wise unstructured magnitude pruning without recovery fine-tuning or sparse-kernel acceleration.
- **Incomplete measurements:** Hard-label DistilBERT lacks median latency, memory, energy, emissions, and checkpoint-size measurements. INT8/INT4 serialized checkpoint sizes are not reported.
- **Pareto specification:** The archive preserves flags but not the original generating code. Verification uses the disclosed objectives and available complete cases.
- **No broad significance claim:** The Phase 12/15 figures visualize recorded point estimates and do not establish statistical significance.

