# Methodology

## Study design

The task is 10-class disaster-tweet classification on HumAID. A QLoRA-fine-tuned Qwen2.5-3B-Instruct model is the common teacher/reference. The study compares two clearly separated deployment scopes.

**Architecture-preserving compression:** FP16 Qwen, INT8 Qwen, INT4 NF4 Qwen, and 30/50/70% unstructured magnitude-pruned Qwen variants.

**Task-specific deployment alternative:** hard-label DistilBERT and knowledge-distilled DistilBERT classifiers.

## Data

The cleaned corpus contains 53,530 tweets. The training split contains 37,471 examples and the held-out test split contains 8,030. Direct cross-method quality comparisons use one fixed, stratified 2,000-example subset generated with `random_state=42`. The missing/found category has support 9 in this subset.

## QLoRA teacher

Qwen2.5-3B-Instruct is adapted with QLoRA. The saved record reports 7,372,800 trainable adapter parameters. The LoRA adapter is merged into the base model before runtime quantization or pruning so the evaluated model contains the task-specific update.

## Quantization

The merged model is evaluated as FP16, INT8, and INT4 NF4. These are inference configurations of the Qwen family. Reported INT8/INT4 checkpoint sizes are missing and remain missing.

## Pruning

Layer-wise unstructured magnitude pruning is applied once at 30%, 50%, and 70% sparsity without recovery fine-tuning. Dense serialization is retained, so sparsity does not imply a smaller stored checkpoint. Exact-label generation failures are recorded as unmatched outputs.

## Knowledge distillation

The teacher contains 3,085,938,688 parameters. The DistilBERT student contains 66,961,162 parameters and a 10-class head. The teacher scores complete candidate label sequences conditioned on the classification prompt; length-normalized scores form a 10-class soft target at temperature 4.0. The student objective combines hard-label cross-entropy and temperature-scaled KL divergence with alpha 0.5. A hard-label-only DistilBERT control isolates the observed incremental value of the soft-target objective in this run.

## Evaluation

Quality metrics are accuracy, macro F1, weighted F1, and unmatched rate. Runtime reporting distinguishes median single-example latency from separately measured throughput. GPU memory is peak allocated memory as recorded by the experiment. Energy and emissions are CodeCarbon estimates on a fixed 100-example workload, not direct power measurements.

## Pareto analysis

The archived mixed-system complete-case analysis includes FP16 Qwen, INT8 Qwen, INT4 NF4 Qwen, pruned-30% Qwen, and distilled DistilBERT. It maximizes macro F1 and minimizes median latency, peak allocated GPU memory, and estimated energy. The hard-label control is excluded because its latency, memory, and energy are missing. The pruned-50% and pruned-70% models are excluded as unusable.

