# Models

Large checkpoints are intentionally excluded from this repository.

The study uses:

- Qwen2.5-3B-Instruct as the teacher/base architecture, fine-tuned with QLoRA and merged before post-training compression.
- DistilBERT-base-uncased with a 10-class classification head as the hard-label control and distilled student architecture.

Do not commit full Qwen weights, merged checkpoints, `.safetensors`, adapter checkpoints, Kaggle caches, or downloaded Hugging Face caches. When release permissions and hosting are settled, add links plus checksums here instead of storing weights in Git.

Before adding any model link, record:

- exact upstream model identifier and revision;
- adapter/checkpoint identifier and checksum;
- license and redistribution constraints;
- training configuration and label mapping;
- intended use and known limitations.

The distilled student is a task-specific deployment alternative. It is not another compressed Qwen checkpoint.

