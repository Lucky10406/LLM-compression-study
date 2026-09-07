# Elevator pitches

## Natural 15-second explanation

We tested three ways to make a fine-tuned 3-billion-parameter language model cheaper for disaster-tweet classification. Four-bit quantization best preserved the Qwen architecture, while a small distilled DistilBERT classifier delivered the strongest measured task-specific deployment trade-off.

## 30-second pitch (under 90 words)

We built an empirical Green AI study around 10-class HumAID disaster-tweet classification. Starting from a QLoRA-fine-tuned Qwen2.5-3B model, We compared INT8 and INT4 quantization, 30–70% unstructured pruning, and knowledge distillation. INT4 NF4 was the strongest Qwen quality-memory option, while naive pruning collapsed between 30% and 50% sparsity. A 66.96M-parameter distilled DistilBERT reached 0.7735 accuracy and 0.7630 macro F1 with about 4.38 ms median latency. The key lesson is that compression must be evaluated as a multi-objective deployment problem.


Authors: Mayuresh Sharma and Aditya Chauhan — Guru Gobind Singh Indraprastha University (GGSIPU), New Delhi.
