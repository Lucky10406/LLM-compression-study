# Data

Processed HumAID tweet CSVs are intentionally not committed. The included preprocessing notebook downloads `QCRI/HumAID-all`, cleans the text, removes duplicate cleaned tweets, and creates stratified train/validation/test splits locally under `data/processed/`.

Before running it, review the current HumAID dataset terms and the Hugging Face dataset card. Do not publish raw or processed tweet text unless redistribution is explicitly permitted.

Expected locally generated files:

```text
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
```

These paths are excluded by `.gitignore`.

