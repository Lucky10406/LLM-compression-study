# Phase 13 evidence audit

This audit accompanies the Phase 13 research report. It distinguishes preservation and metric-level verification from experimental replication. The Phase 12 DOCX was edited directly. The supplied Phase 10 PDF supplies historical provenance; Phase 11 and Phase 12 backups supply the new saved results. No training, inference, energy measurement, or new inferential statistical test was performed for this report.

## Evidence inventory before editing

No notebooks, executable experiment scripts, configuration JSON files, raw predictions, checkpoint files, split CSVs, raw CodeCarbon logs, or environment lockfiles were found in the two supplied backups. Historical descriptions of those artifacts in the Phase 10 manuscript are retained but do not establish their availability here. The attached Phase 13 specification controls report preparation; it is not experimental evidence.

| Inspected source | Bytes | SHA-256 |
| --- | ---: | --- |
| `output/research_paper_through_phase12.docx` | 3708996 | `c276f3b28b3004dbed5c4f9ecdb396753bf8e4c31a455218e656dd4501e9e3d2` |
| `upload/research_paper_through_phase10(3).pdf` | 2062943 | `48e6e814502ec5fd6d3327f28c4c8bd07a0b7c6ecc60b8af5af7e69a66d5497c` |
| `source/phase11/PHASE11_SUMMARY.txt` | 1676 | `a8d0039337d517ad69ca56bae9b866f423dad450af65b474357f44abd1cbb2b1` |
| `source/phase11/master_comparison.csv` | 1334 | `8f8c53295ab69e5e61c192e5e38fbef794c2b0b3c202e87d1b3bd5db5c1d74ed` |
| `source/phase11/pareto_analysis.csv` | 952 | `c58ff32bd871c6d2aeaf0e8d4420dfc23c2828eb6554b1d6c7cf0f8b9df89f58` |
| `source/phase11/rare_class_comparison.csv` | 153 | `e1c7834c8ce0629eae4beae9c68059e6a0fcf8de4293e291820c944d67d509f1` |
| `source/phase11/usable_model_deltas.csv` | 1473 | `9a28519da5874ac96d2843fbdc2165b4e7a3347b47c86eaa6905b1548e6b4751` |
| `source/phase12/phase12_plot_data.csv` | 1186 | `3dc04300cce628f880d557185176d7b8b6c0513a566892f9ac3f8f7ba3dde177` |
| `source/phase12/plots/01_performance_vs_latency.pdf` | 20590 | `e6ae2d8d629a0e462e0f6e326c2d46001065493054f783047fbada3cce1070b7` |
| `source/phase12/plots/01_performance_vs_latency.png` | 218141 | `f02d90e86547a65f6b62eca171d4b515ac4637fd8097408db9279543a84a0a72` |
| `source/phase12/plots/02_energy_comparison.pdf` | 19970 | `e0f74eec12e12dd6cd5f054f1a51669c92e587bc2860b08c9cc8bb16486928cc` |
| `source/phase12/plots/02_energy_comparison.png` | 204576 | `e76249121706c3566526290f5677fef937240636bd7e035308549e0e21887734` |
| `source/phase12/plots/03_size_memory_comparison.pdf` | 19863 | `7737ca64fb1f22d263bf8a44817b07cc0fd357cc4f480cca4b94d307abfeea23` |
| `source/phase12/plots/03_size_memory_comparison.png` | 289887 | `12dcb509067ab085b294ec2a30ca0a1a191d179be263b74fbd6e1e020223a92e` |
| `source/phase12/plots/plot_summary.md` | 3935 | `a50d7363fb45988bc6d3bec6c649769977fdd0996fa7802a4e61c0510ae6231c` |
| `upload/phase11-final-backup(2).zip` | 3650 | `e0a2f6cbf6b6b523f5f541b08159203fa1ca213f7bd0735d05a7d580c78ae9fd` |
| `upload/phase12-final-backup(2).zip` | 597267 | `60e8d1495fedfae606a2ef451daaad0d2b6a6c130e36691251ce9e01a46fb5d4` |
| `upload/Pasted markdown(6).md` | 20723 | `7f3468fa61db4887cac00ae56200f7a9bff01f8662ac8d5d2c31fbd303158b09` |

## Evidence classification

“Historical reported measurement” means the supplied Phase 10 paper records the result; the original experiment could not be re-executed or checked against raw observations. “Saved metric” means a Phase 11/12 CSV value was checked, not that its underlying predictions were independently validated. Classification scores are derived from predictions; hardware timing, memory, and storage are measured quantities as reported; CodeCarbon energy and emissions are modeled estimates. Percentage changes, unit conversions, and Pareto verification are derived analyses. Missing values remain missing.

## Claim-to-evidence map

| Claim or method | Source | Report location | Type and verification boundary |
| --- | --- | --- | --- |
| 53,531 loaded; one duplicate removed; 53,530 unique; seed 42; 37,471/8,029/8,030 split | Phase 10 PDF §4 | §4; Figure 1; §6.1 | Historical preparation record; counts internally consistent; split files unavailable |
| Training class counts 10,424; 5,951; 4,374; 4,000; 3,577; 3,085; 2,642; 1,960; 1,283; 175 | Phase 10 Figure 1 | §4; Figure 1 | Historical figure; counts sum to 37,471; no new distribution estimated |
| UTF-8 cleanup, leading retweet-marker removal, exact deduplication, stratification | Phase 10 §4 | §4 | Historical method; no event-level or near-duplicate exclusion claim |
| TF-IDF 5,000 unigram/bigram; weighted logistic regression; max_iter 1,000; validation macro/weighted F1 0.7032/0.7314; test 71.30%, 0.6965, 0.7168 | Phase 10 §5.1 | §5.1; §7.1; RQ1 answer | Historical method and classification metrics; validation and test remain separate |
| QLoRA r16, alpha32, dropout0.05, q/k/v/o projections, 288 tensors, 7,372,800 adapters; 3 epochs, batch4, accumulation4, LR2e-4, FP16 | Phase 10 §5.2 and protocol table | §5.2; Table 1; Figure 2 | Historical configuration; frozen four-bit base and adapter merging distinguished from inference quantization; unspecified numerical settings remain unreported |
| Qwen full test 5,929/8,030 correct, 73.84% accuracy, 0.7205 macro F1, 0.7288 weighted F1; four unmatched, 0.050% | Phase 10 §7.1 and Figure 3 | §7.1; Figure 3; abstract; conclusion | Historical classification metrics; full-test denominator not confused with benchmark |
| Fixed stratified 2,000-example subset, random_state42; exact label parser, deterministic new-token generation | Phase 10 evaluation protocol | §6.1; Table 1 | Historical protocol; sample files absent; exact prompt/template/version not reconstructed |
| INT8 and NF4 Double Quantization runtime loading; no independently measured quantized disk checkpoint | Phase 10 §6 | §5.6; §6.1; Tables 3, 13–15; Figure 17 | Historical configuration and missingness; bit width not used to invent storage |
| Exact paired McNemar: FP16-only/INT8-only 39/19, p0.0119; FP16-only/INT4-only 51/82, p0.0090 | Phase 10 §7.2 | §7.2; §6.2; RQ2 answer | Retained previously performed tests; not new tests; raw paired predictions absent |
| Single-example warm-up5, trials50, CUDA synchronization; teacher median, student mean4.40 ms/median4.38 ms; separate full-run throughput | Phase 10 protocol and §§7.3, 7.8 | §6.1; Tables 1, 3, 8 | Historical measured runtime; median reciprocal not substituted for throughput |
| Layer-wise one-shot magnitude zeroing at30/50/70%; direct zeroing due mask-memory failure; dense serialization; independent starting checkpoints | Phase 10 §5.3 | §5.3; Tables 5–6; Figures 9–11 | Historical method; failure retained; no recovery training claimed |
| 30% prune peak2.8954 GB is busiest GPU, model spans two GPUs; dense size5.763 GB unchanged; 50/70 disk sizes absent | Phase 10 §§5.3, 7.6; master CSV | §6.1; §7.6; Tables 6, 16–18 | Reported measurement; cross-device caveat excludes total-memory interpretation |
| Teacher3,085,938,688; student66,961,162; teacher full-label mean logprob targets, T4, alpha0.5, T² KL+CE; matrix37,471×10, noNaNs, normalized | Phase 10 §§5.4, 7.7 | §5.4; Tables 7–8; Figures 12, 14 | Historical training record; teacher fixed during offline-target student optimization; no code-level revalidation |
| Student3epochs, batch16, LR2e-5, maxlen128, clip1; matched hard-label control; +0.95pp accuracy, +0.01627 macro F1, +0.01646 weighted F1 | Phase 10; master CSV; Phase11 summary | §5.4; §7.7; RQ5; Tables 7, 19–20 | Historical configuration and saved metrics; differences derived; single-run limitation |
| All eight models, including Pruned_50/70 failures and incomplete HardLabel_DistilBERT | master_comparison.csv | Tables 13–20; main Tables 2–8 | Saved source strings preserved; every master field checked; no unavailable fields imputed |
| FP16-relative accuracy/F1/memory/energy changes for six usable models | usable_model_deltas.csv | §5.5; Table 10; Tables 13–16, 19–20 | Derived; negative resource reduction denotes increase; cross-architecture scope stated |
| Archived five-model dominance flags; student sole non-dominated complete usable system | pareto_analysis.csv | §5.5; §7.9; Table 9; Tables 13–16, 19 | Archived derived flags; independently reproduced with stated four objectives; original code/objective specification absent |
| Qwen FP16/INT4 projection frontier; Pruned_30 additionally survives four-objective frontier; INT8 dominated by INT4 | master metrics plus stated objective definition | §7.9; Table 9; RQ6 | New scoped derivation from saved metrics; per-GPU confound explicitly retained |
| Rare F1 rows: .88/.51, .88/.49, .88/.52, .94/.50, .84/.55; missing/found support9 | rare_class_comparison.csv; both summaries | §5.5; §7.10; Table 11; Tables 13–16, 19 | Saved two-decimal F1; support from summaries; urgent support and confusion counts unavailable |
| Performance-latency plot and99.44% student median-latency reduction | phase12_plot_data.csv; 01_performance_vs_latency PNG/PDF | Figure 15; §7.11 | Supplied figure unchanged; ms derived from seconds; scope permits task-specific classifier |
| Energy plot,0.141 versus26.089 Wh/1,000;99.46% student estimated-energy reduction | phase12_plot_data.csv; 02_energy_comparison PNG/PDF | Figure 16; §7.11 | Modeled estimate normalized from100 cases; no new1,000-case energy experiment |
| Size/memory plot; student95.66% storage and95.35% memory reductions | phase12_plot_data.csv; 03_size_memory_comparison PNG/PDF | Figure 17; §7.11 | Derived relative changes; GB×1,024 source convention retained without byte-unit guarantee |
| Energy and emissions values across precision, pruning, student; fixed100-case runtimes | Phase 10 Tables 4,6,8; master CSV | §§6.1,7.4,7.6,7.8; Tables 4,6,8,13–20 | CodeCarbon estimates; direct meter, exact configuration and raw logs absent |
| Four-bit quality preserved;30% prune loss10.15pp; collapse between tested30% and50% | master CSV; historical prune results | §8.8; conclusion | Descriptive threshold interpretation; no practical tolerance prespecified and no universal compression threshold claimed |
| Phase1–10 historical artifact list, including approximately474MB Phase10 archive | Phase 10 reproducibility section | §10 | Historical inventory only; those artifacts were not supplied or inspected here |
| Green AI, HumAID, Qwen, LoRA, QLoRA, LLM.int8, CodeCarbon, distillation background | References [1]–[8] | §3 and cited methods/discussion; bibliography | Bibliographic metadata/official links checked; documentation access does not establish experimental software version |

## Full numeric table ledger

Every cell below is copied from the final DOCX, whose 20 table bodies were programmatically compared with the Phase 12 DOCX and found identical. Main Tables 1–8 preserve the historical manuscript values and display precision. Tables 9–12 preserve the Phase 12 synthesis. Former appendix Tables C1–C8 are renumbered 13–20 without cell edits. In those records, original field names retain the distinction between measured, estimated, and derived quantities. “Not reported” and “not included” are not zero.

### Table 1

Source: Phase 10 manuscript. Report location: §6.

| Component | Setting |
| --- | --- |
| Base model | Qwen2.5-3B-Instruct |
| Task | 10-class HumAID tweet classification |
| LoRA | r=16, alpha=32, dropout=0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Fine-tuning | 3 epochs; batch 4; grad accumulation 4; LR 2e-4 |
| Direct quality benchmark | 2,000 stratified test examples; random_state=42 |
| Generation | Deterministic; exact-label prompt; max_new_tokens <= 15 |
| Latency | 5 warm-up + 50 timed trials; CUDA synchronized |
| Energy | 100 fixed examples; CodeCarbon estimate |
| Hardware | 2 x NVIDIA Tesla T4 available; model execution primarily on GPU 0 |
| Pruning | Layer-wise unstructured magnitude pruning at 30%, 50%, 70%; fresh FP16 source per level |
| Distillation student | DistilBERT-base-uncased; 66,961,162 parameters; 10-class head |
| Distillation training | 3 epochs; batch 16; LR 2e-5; max length 128; T=4.0; alpha=0.5 |
| Teacher soft targets | Full-label conditional sequence scoring; length- normalized; 37,471 x 10 saved matrix |
| Student ablation | Matched DistilBERT trained with hard-label cross-entropy only |

### Table 2

Source: Phase 10 manuscript. Report location: §7.

| Model | Accuracy | Macro P | Macro R | Macro F1 | Weighted F1 |
| --- | --- | --- | --- | --- | --- |
| FP16 baseline | 0.7410 | 0.7504 | 0.7493 | 0.7325 | 0.7311 |
| INT8 | 0.7310 | 0.7481 | 0.7346 | 0.7232 | 0.7232 |
| INT4 NF4 | 0.7565 | 0.7604 | 0.7649 | 0.7497 | 0.7490 |

### Table 3

Source: Phase 10 manuscript. Report location: §7.

| Model | Median latency (s) | Latency throughput (/s) | Peak GPU memory (GB) | 2k run throughput (/s) |
| --- | --- | --- | --- | --- |
| FP16 baseline | 0.782 | 1.271 | 5.769 | 0.976 |
| INT8 | 2.849 | 0.351 | 3.245 | 0.318 |
| INT4 NF4 | 1.074 | 0.929 | 1.975 | 0.760 |

### Table 4

Source: Phase 10 manuscript. Report location: §7.

| Model | 100-sample runtime (s) | Estimated energy (kWh) | Estimated emissions (g CO2e) |
| --- | --- | --- | --- |
| FP16 baseline | 84.40 | 0.002609 | 1.181 |
| INT8 | 298.37 | 0.007451 | 3.372 |
| INT4 NF4 | 111.31 | 0.003439 | 1.556 |

### Table 5

Source: Phase 10 manuscript. Report location: §7.

| Variant | Sparsity | Accuracy | Macro P | Macro R | Macro F1 | Unmatched |
| --- | --- | --- | --- | --- | --- | --- |
| FP16 reference | 0% | 0.7410 | 0.7504 | 0.7493 | 0.7325 | - |
| Pruned 30% | 30% | 0.6395 | 0.7674 | 0.6483 | 0.6333 | 6 / 2000 |
| Pruned 50% | 50% | 0.0005 | 0.0500 | 0.0007 | 0.0014 | 1992 / 2000 |
| Pruned 70% | 70% | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2000 / 2000 |

### Table 6

Source: Phase 10 manuscript. Report location: §7.

| Variant | Median latency (s) | Throughput (/s) | Peak allocated GPU (GB) | 100-sample runtime (s) | Energy (kWh) | CO2e (g) |
| --- | --- | --- | --- | --- | --- | --- |
| FP16 reference | 0.782 | 1.271 | 5.769 | 84.40 | 0.002609 | 1.181 |
| Pruned 30% | 1.002 | 0.995 | 2.895* | 101.98 | 0.002975 | 1.346 |
| Pruned 50% | 0.754 | 1.325 | 5.769* | 76.96 | 0.002516 | 1.139 |
| Pruned 70% | 0.782 | 1.276 | 5.827* | 74.53 | 0.002447 | 1.107 |

### Table 7

Source: Phase 10 manuscript. Report location: §7.

| Variant | Parameters | Accuracy | Macro F1 | Weighted F1 | Unmatched |
| --- | --- | --- | --- | --- | --- |
| FP16 Qwen teacher | 3.086B | 0.7410 | 0.7325 | 0.7311 | - |
| INT8 Qwen | 3.086B | 0.7310 | 0.7232 | 0.7232 | - |
| INT4 NF4 Qwen | 3.086B | 0.7565 | 0.7497 | 0.7490 | - |
| Pruned 30% Qwen | 3.086B dense | 0.6395 | 0.6333 | 0.6318 | 6 |
| Pruned 50% Qwen | 3.086B dense | 0.0005 | 0.0014 | 0.0010 | 1992 |
| Pruned 70% Qwen | 3.086B dense | 0.0000 | 0.0000 | 0.0000 | 2000 |
| Hard-label DistilBERT | 66.96M | 0.7640 | 0.7467 | 0.7551 | 0 |
| Distilled DistilBERT | 66.96M | 0.7735 | 0.7630 | 0.7716 | 0 |

### Table 8

Source: Phase 10 manuscript. Report location: §7.

| Metric | FP16 Qwen teacher | Distilled DistilBERT |
| --- | --- | --- |
| Parameters | 3,085,938,688 | 66,961,162 |
| Benchmark accuracy | 0.7410 | 0.7735 |
| Macro F1 | 0.7325 | 0.7630 |
| Checkpoint size | 5.763 GB | 0.2501 GB (256.13 MB) |
| Peak allocated GPU memory | 5.769 GB | 0.2681 GB |
| Single-example latency | 0.782 s median | 0.00440 s mean; 0.00438 s median |
| Formal throughput | 1.271 /s | 227.36 /s |
| 2k benchmark throughput | 0.976 /s | 220.72 /s |
| 100-sample runtime | 84.40 s | 0.497 s |
| Estimated energy | 0.002609 kWh | 1.413e-5 kWh |
| Estimated CO2e | 1.181 g | 0.0064 g |

### Table 9

Source: Phase 11 CSVs and explicit scoped Pareto derivation. Report location: §7.

| Variant | Archived mixed-system flag | Qwen only: F1 + latency | Qwen only: F1 + latency + memory + energy |
| --- | --- | --- | --- |
| FP16 Qwen | Dominated | Non-dominated | Non-dominated |
| INT8 Qwen | Dominated | Dominated | Dominated |
| INT4 NF4 Qwen | Dominated | Non-dominated | Non-dominated |
| Pruned 30% | Dominated | Dominated | Non-dominated* |
| Distilled DistilBERT | Non-dominated | Outside scope | Outside scope |

### Table 10

Source: usable_model_deltas.csv. Report location: §7.

| Variant | Accuracy Δ (pp) | Macro F1 Δ | Memory reduction (%) | Energy reduction (%) |
| --- | --- | --- | --- | --- |
| FP16_Qwen | +0.00 | +0.00000 | 0.00 | 0.00 |
| INT8_Qwen | -1.00 | -0.00931 | 43.75 | -185.60 |
| INT4_NF4_Qwen | +1.55 | +0.01719 | 65.77 | -31.81 |
| Pruned_30 | -10.15 | -0.09920 | 49.81 | -14.02 |
| Distilled_DistilBERT | +3.25 | +0.03046 | 95.35 | 99.46 |
| HardLabel_DistilBERT | +2.30 | +0.01419 | Not reported | Not reported |

### Table 11

Source: rare_class_comparison.csv. Report location: §7.

| Variant | Missing/found F1 | Urgent-needs F1 |
| --- | --- | --- |
| FP16_Qwen | 0.88 | 0.51 |
| INT8_Qwen | 0.88 | 0.49 |
| INT4_NF4_Qwen | 0.88 | 0.52 |
| Pruned_30 | 0.94 | 0.5 |
| Distilled_DistilBERT | 0.84 | 0.55 |

### Table 12

Source: Historical stages plus supplied Phase 11/12 completion evidence. Report location: §10.

| Stage | Status through Phase 12 | Retained or integrated evidence |
| --- | --- | --- |
| Magnitude pruning (30/50/70%) | Completed | Quality, latency, throughput, memory, dense-size limitation, energy, CO2e |
| Knowledge-distilled student | Completed | Student size, hard-label ablation, quality, latency, memory, checkpoint size, energy, CO2e |
| Final Pareto comparison (Phase 11) | Completed | Common metric record, usable deltas, Pareto flags, rare-class findings; separate comparison scopes |
| Visualizations (Phase 12) | Completed | Performance-latency, energy, and size-memory figures; exact plot-data record |

### Table 13

Source: master_comparison.csv plus all model-matched delta, rare-class, Pareto and Phase 12 plot fields. Report location: Appendix C.

| Source field | Recorded value |
| --- | --- |
| family | baseline |
| usable | True |
| accuracy | 0.741 |
| macro_f1 | 0.732505725427883 |
| weighted_f1 | 0.7310994656252274 |
| median_latency_s | 0.782 |
| throughput_sps | 1.271 |
| peak_gpu_memory_gb | 5.769 |
| energy_100_kwh | 0.0026088649956977 |
| co2_100_g | 1.1808264965973 |
| checkpoint_size_gb | 5.763 |
| parameters | 3085938688 |
| unmatched_rate | 0.0 |
| dominated (archived mixed-system scope) | True |
| accuracy_delta_pp | 0.0 |
| macro_f1_delta | 0.0 |
| memory_reduction_pct | 0.0 |
| energy_reduction_pct | 0.0 |
| missing_found_f1 | 0.88 |
| urgent_needs_f1 | 0.51 |
| latency_ms | 782.0 |
| peak_gpu_memory_mb | 5907.456 |
| checkpoint_size_mb | 5901.312 |
| energy_wh_per_1000 | 26.088649956977 |

### Table 14

Source: master_comparison.csv plus all model-matched delta, rare-class, Pareto and Phase 12 plot fields. Report location: Appendix C.

| Source field | Recorded value |
| --- | --- |
| family | quantization |
| usable | True |
| accuracy | 0.731 |
| macro_f1 | 0.7232 |
| weighted_f1 | 0.7232 |
| median_latency_s | 2.849 |
| throughput_sps | 0.351 |
| peak_gpu_memory_gb | 3.245 |
| energy_100_kwh | 0.0074510088901041 |
| co2_100_g | 3.3724814194397 |
| checkpoint_size_gb | Not reported |
| parameters | 3085938688 |
| unmatched_rate | 0.0 |
| dominated (archived mixed-system scope) | True |
| accuracy_delta_pp | -1.0000000000000009 |
| macro_f1_delta | -0.009305725427882994 |
| memory_reduction_pct | 43.7510833766684 |
| energy_reduction_pct | -185.6034674999135 |
| missing_found_f1 | 0.88 |
| urgent_needs_f1 | 0.49 |
| latency_ms | 2849.0 |
| peak_gpu_memory_mb | 3322.88 |
| checkpoint_size_mb | Not reported |
| energy_wh_per_1000 | 74.510088901041 |

### Table 15

Source: master_comparison.csv plus all model-matched delta, rare-class, Pareto and Phase 12 plot fields. Report location: Appendix C.

| Source field | Recorded value |
| --- | --- |
| family | quantization |
| usable | True |
| accuracy | 0.7565 |
| macro_f1 | 0.7497 |
| weighted_f1 | 0.749 |
| median_latency_s | 1.074 |
| throughput_sps | 0.929 |
| peak_gpu_memory_gb | 1.975 |
| energy_100_kwh | 0.0034387051172414 |
| co2_100_g | 1.5564293756555 |
| checkpoint_size_gb | Not reported |
| parameters | 3085938688 |
| unmatched_rate | 0.0 |
| dominated (archived mixed-system scope) | True |
| accuracy_delta_pp | 1.5499999999999958 |
| macro_f1_delta | 0.017194274572117085 |
| memory_reduction_pct | 65.76529727855781 |
| energy_reduction_pct | -31.80847314491919 |
| missing_found_f1 | 0.88 |
| urgent_needs_f1 | 0.52 |
| latency_ms | 1074.0 |
| peak_gpu_memory_mb | 2022.4 |
| checkpoint_size_mb | Not reported |
| energy_wh_per_1000 | 34.387051172414 |

### Table 16

Source: master_comparison.csv plus all model-matched delta, rare-class, Pareto and Phase 12 plot fields. Report location: Appendix C.

| Source field | Recorded value |
| --- | --- |
| family | pruning |
| usable | True |
| accuracy | 0.6395 |
| macro_f1 | 0.6333022599482043 |
| weighted_f1 | 0.6318104752816464 |
| median_latency_s | 1.001759561999961 |
| throughput_sps | 0.9949162372233542 |
| peak_gpu_memory_gb | 2.8954 |
| energy_100_kwh | 0.002974726862353 |
| co2_100_g | 1.3464231782782 |
| checkpoint_size_gb | 5.763 |
| parameters | 3085938688 |
| unmatched_rate | 0.003 |
| dominated (archived mixed-system scope) | True |
| accuracy_delta_pp | -10.150000000000004 |
| macro_f1_delta | -0.09920346547967862 |
| memory_reduction_pct | 49.81105910903103 |
| energy_reduction_pct | -14.023794533586287 |
| missing_found_f1 | 0.94 |
| urgent_needs_f1 | 0.5 |
| latency_ms | 1001.7595619999611 |
| peak_gpu_memory_mb | 2964.8896 |
| checkpoint_size_mb | 5901.312 |
| energy_wh_per_1000 | 29.747268623529994 |

### Table 17

Source: master_comparison.csv plus all model-matched delta, rare-class, Pareto and Phase 12 plot fields. Report location: Appendix C.

| Source field | Recorded value |
| --- | --- |
| family | pruning |
| usable | False |
| accuracy | 0.0005 |
| macro_f1 | 0.0013986013986013986 |
| weighted_f1 | 0.000986013986013986 |
| median_latency_s | 0.7536016564999954 |
| throughput_sps | 1.3254230991334082 |
| peak_gpu_memory_gb | 5.7688 |
| energy_100_kwh | 0.0025160002320851 |
| co2_100_g | 1.1387939753075 |
| checkpoint_size_gb | Not reported |
| parameters | 3085938688 |
| unmatched_rate | 0.996 |
| dominated (archived mixed-system scope) | Not included |
| accuracy_delta_pp | Not reported |
| macro_f1_delta | Not reported |
| memory_reduction_pct | Not reported |
| energy_reduction_pct | Not reported |
| missing_found_f1 | Not reported |
| urgent_needs_f1 | Not reported |
| latency_ms | 753.6016564999954 |
| peak_gpu_memory_mb | 5907.2512 |
| checkpoint_size_mb | Not reported |
| energy_wh_per_1000 | 25.160002320851 |

### Table 18

Source: master_comparison.csv plus all model-matched delta, rare-class, Pareto and Phase 12 plot fields. Report location: Appendix C.

| Source field | Recorded value |
| --- | --- |
| family | pruning |
| usable | False |
| accuracy | 0.0 |
| macro_f1 | 0.0 |
| weighted_f1 | 0.0 |
| median_latency_s | 0.7824434344993279 |
| throughput_sps | 1.276216289105717 |
| peak_gpu_memory_gb | 5.8273 |
| energy_100_kwh | 0.0024468577306169 |
| co2_100_g | 1.1074986426976 |
| checkpoint_size_gb | Not reported |
| parameters | 3085938688 |
| unmatched_rate | 1.0 |
| dominated (archived mixed-system scope) | Not included |
| accuracy_delta_pp | Not reported |
| macro_f1_delta | Not reported |
| memory_reduction_pct | Not reported |
| energy_reduction_pct | Not reported |
| missing_found_f1 | Not reported |
| urgent_needs_f1 | Not reported |
| latency_ms | 782.4434344993279 |
| peak_gpu_memory_mb | 5967.1552 |
| checkpoint_size_mb | Not reported |
| energy_wh_per_1000 | 24.468577306169 |

### Table 19

Source: master_comparison.csv plus all model-matched delta, rare-class, Pareto and Phase 12 plot fields. Report location: Appendix C.

| Source field | Recorded value |
| --- | --- |
| family | distillation |
| usable | True |
| accuracy | 0.7735 |
| macro_f1 | 0.7629680730982821 |
| weighted_f1 | 0.7715770921582532 |
| median_latency_s | 0.004378193500087946 |
| throughput_sps | 227.36370581025434 |
| peak_gpu_memory_gb | 0.2681 |
| energy_100_kwh | 1.4132845669886998e-05 |
| co2_100_g | 0.006396819562087144 |
| checkpoint_size_gb | 0.250126 |
| parameters | 66961162 |
| unmatched_rate | 0.0 |
| dominated (archived mixed-system scope) | False |
| accuracy_delta_pp | 3.2499999999999973 |
| macro_f1_delta | 0.030462347670399126 |
| memory_reduction_pct | 95.35274744323107 |
| energy_reduction_pct | 99.45827608200526 |
| missing_found_f1 | 0.84 |
| urgent_needs_f1 | 0.55 |
| latency_ms | 4.3781935000878995 |
| peak_gpu_memory_mb | 274.5344 |
| checkpoint_size_mb | 256.129024 |
| energy_wh_per_1000 | 0.14132845669886998 |

### Table 20

Source: master_comparison.csv plus all model-matched delta, rare-class, Pareto and Phase 12 plot fields. Report location: Appendix C.

| Source field | Recorded value |
| --- | --- |
| family | ablation |
| usable | True |
| accuracy | 0.764 |
| macro_f1 | 0.7466958537197546 |
| weighted_f1 | 0.7551154514211491 |
| median_latency_s | Not reported |
| throughput_sps | 223.4243109001946 |
| peak_gpu_memory_gb | Not reported |
| energy_100_kwh | Not reported |
| co2_100_g | Not reported |
| checkpoint_size_gb | Not reported |
| parameters | 66961162 |
| unmatched_rate | 0.0 |
| dominated (archived mixed-system scope) | Not included |
| accuracy_delta_pp | 2.300000000000002 |
| macro_f1_delta | 0.014190128291871607 |
| memory_reduction_pct | Not reported |
| energy_reduction_pct | Not reported |
| missing_found_f1 | Not reported |
| urgent_needs_f1 | Not reported |
| latency_ms | Not reported |
| peak_gpu_memory_mb | Not reported |
| checkpoint_size_mb | Not reported |
| energy_wh_per_1000 | Not reported |

## Figure preservation and provenance

All 17 image payloads have identical SHA-256 digests to their counterparts in the Phase 12 DOCX. Figures 1–14 are inherited from the Phase 10 paper; Figures 15–17 are the supplied Phase 12 PNGs. Captions provide interpretive scope and correct known label issues while preserving the assets.

| Figure | Caption / content | Source | Image SHA-256 |
| --- | --- | --- | --- |
| 1 | Figure 1. Class distribution of the 37,471-example training split. The dataset is substantially imbalanced; the rarest class, missing/found people, has only 175 training examples. | Phase 10 figure 1 | `9b4439a70e3eac42f1f56a6671c1c8a2a5f42e6d2568fb35f735dc986b172ad3` |
| 2 | Figure 2. End-to-end experimental pipeline through knowledge distillation. Quantization and pruning operate on the merged FP16 teacher, whereas distillation trains a separate 66.96M-parameter DistilBERT classifier from teacher-derived soft targets and ground-truth labels; all variants feed a shared evaluation framework where the architecture permits a fair comparison. The original Phase 1-10 pipeline is preserved; Sections 5.5 and 7.9-7.11 describe the completed synthesis and visualization stages. | Phase 10 figure 2 | `a87f027991bf4d00a9c9ce532c80dc2c9768b0e35e3ad6d1495e7c6073e0a2a4` |
| 3 | Figure 3. Row-normalized confusion matrix for the QLoRA fine-tuned model on the full 8,030-example test set. The most persistent difficulty is the broad “other relevant information” category, which overlaps semantically with several actionable categories. | Phase 10 figure 3 | `5e982a5aac3e23cf2490ecc78d0501a918f31976da5bbf961f3899446ba3f25a` |
| 4 | Figure 4. Accuracy, macro F1, and weighted F1 for FP16, INT8, and INT4 NF4 on the same 2,000 test examples. | Phase 10 figure 4 | `d3294e49c335dc6175ef19160aa504587c89df88ff3785f52496d4a43170d828` |
| 5 | Figure 5. Per-class F1 across precision variants. INT4 gains are concentrated in several categories, including caution/advice, infrastructure damage, and other relevant information, while sympathy/support and not-humanitarian do not improve. | Phase 10 figure 5 | `f81c2555f6a28a7fbea4c9a961296b749fbf55aa0c362f4c9432b88eca5d6eeb` |
| 6 | Figure 6. Lower precision substantially reduces peak allocated GPU memory, but the Tesla T4 runtime does not convert this memory reduction into lower latency. | Phase 10 figure 6 | `cb1792939ae25fab0ab698b5f83898250ff2abc34ca08c0cc29aa330b9acc549` |
| 7 | Figure 7. Accuracy-memory trade-off. Under these measurements INT4 NF4 provides the strongest memory/quality point, whereas FP16 remains preferable when latency and energy dominate. | Phase 10 figure 7 | `d7d1828e404a956a4be232d7c980ed336e2945b5aaf6be5a8b9b748d1c6ebe98` |
| 8 | Figure 8. Estimated operational energy and carbon emissions for the same 100-example workload. INT8 is the least efficient variant under the tested T4/bitsandbytes configuration. | Phase 10 figure 8 | `14116e57aaa65b584d3989467177a7435ed4278058bfce5dc2bd6065414e3c17` |
| 9 | Figure 9. Accuracy and macro F1 versus unstructured sparsity. The degradation is strongly nonlinear, with a functional cliff between 30% and 50% sparsity. | Phase 10 figure 9 | `931c5d1b69ad76c935d0a6dba9a4eba9c8a412685f9e99d545949eeb140a7919` |
| 10 | Figure 10. Invalid/unmatched generation rate across the pruning sweep. Generation-format failure rises from 0.3% at 30% sparsity to 99.6% at 50% and 100% at 70%. | Phase 10 figure 10 | `35346beb5ffd181012a2e6efc02f4bf17a3d1f342df204d3310d315e620b3219` |
| 11 | Figure 11. Accuracy versus estimated energy for the precision and pruning variants. The low-energy 50% and 70% pruning points are unusable because their classification behavior has collapsed; efficiency cannot be interpreted independently of quality. | Phase 10 figure 11 | `0c1d750b37d68a2a1e4750439f8801c42bc00ece69b963f7f6acd3fa0c0d9627` |
| 12 | Figure 12. Accuracy and macro F1 across quantization, pruning, and DistilBERT variants on the same fixed 2,000-example benchmark. The distilled student achieves the highest observed accuracy and macro F1, while one-shot pruning collapses beyond 30% sparsity. | Phase 10 figure 12 | `eb3e5d9d3c0504b11c2a13b7a623b3c3f0c334dbf19f4d3bf4993c92490e347f` |
| 13 | Figure 13. Deployment footprint of the FP16 Qwen teacher and distilled DistilBERT student. The log scale highlights reductions in parameter count, checkpoint size, allocated GPU memory, latency, and estimated energy; the comparison is end-to-end and includes the architectural shift from generation to classification. The inherited energy-axis label reads mWh, but the plotted energy values correspond to Wh per 100 examples (approximately 2.609 Wh for FP16 and 0.01413 Wh for the student). Tables 4 and 8 preserve the recorded kWh values; this caption clarifies the original unit label without altering the figure. | Phase 10 figure 13 | `cad1659d68a8a6cc8951ed46f875236cfb12a95f328ac21194849ed145c83618` |
| 14 | Figure 14. Accuracy-energy trade-off including the distilled student. The student occupies a distinct low-energy, high-accuracy region in this task-specific experiment, although its CodeCarbon estimate comes from a very short workload and should be interpreted cautiously. | Phase 10 figure 14 | `5c80d8f9042c67c9a1439085388bdceb26711eaf6d89802e4314c51cb7be996f` |
| 15 | Figure 15. Phase 12 performance versus median latency (logarithmic x-axis): accuracy at left, macro F1 at right. Five usable variants with measured median latency are shown. Distilled DistilBERT is the non-dominated task-specific point; the Qwen-only performance-latency frontier contains FP16 and INT4 NF4. The architectures use different inference procedures. | 01_performance_vs_latency.png | `f02d90e86547a65f6b62eca171d4b515ac4637fd8097408db9279543a84a0a72` |
| 16 | Figure 16. Phase 12 energy estimates normalized to Wh per 1,000 predictions from the same 100-example workload; logarithmic y-axis. Usable variants are shown. This is a scaling of CodeCarbon estimates, not a new workload measurement or direct power-meter reading. Cross-architecture and device-placement caveats apply. | 02_energy_comparison.png | `e76249121706c3566526290f5677fef937240636bd7e035308549e0e21887734` |
| 17 | Figure 17. Phase 12 checkpoint size (left) and peak allocated GPU memory (right). Original MB labels use the source GB ×1,024 convention. FP16 and Pruned 30% have equal dense checkpoint size; quantized disk sizes were not measured. Pruned 30% memory is the busiest-GPU peak in a two-GPU run, not total GPU memory. The DistilBERT comparison is task-specific. | 03_size_memory_comparison.png | `12dcb509067ab085b294ec2a30ca0a1a191d179be263b74fbd6e1e020223a92e` |

## Inconsistencies resolved or explicitly retained

- The original Figure 13 labels energy as mWh, while its plotted values correspond to Wh per 100 examples. The existing corrective caption is retained. The image was not silently altered.

- GB and MB labels in the Phase 12 figures use a ×1,024 conversion. The report preserves the supplied convention and explicitly avoids asserting a verified decimal-byte definition. Model storage and runtime memory remain separate.

- The 30%-pruned memory figure is a per-device peak under two-GPU placement. It is not a verified total-memory reduction; its effect on the four-objective frontier is explicitly qualified.

- The Phase 12 energy summary ranking is valid for usable Qwen variants. Collapsed 50%/70% models have lower numerical energy estimates but are not useful classifiers; they remain in the complete record.

- FP16 is the latency/energy leader among usable Qwen precision variants. “INT4 strongest” is restricted to the observed Qwen quality–memory trade-off, not every resource dimension.

- The archived Pareto flags do not include objective code. Reproduction here uses maximize macro F1 and minimize median latency, recorded peak allocated GPU memory, and estimated energy. This is a disclosed verification specification, not a claim to recover the original algorithm.

- DistilBERT dominance belongs to task-specific deployment comparisons. It does not represent quantization of Qwen or preservation of generative capabilities.

- Hard-label throughput is retained at source precision, but no median latency, memory, energy, carbon, or checkpoint size is reconstructed from it.

- Rounded values in historical text and main tables are retained alongside source-precision records. Minor Phase 12 floating serialization differences in repeated fields are not distinct measured results; additional plotting fields remain in Tables 13–20.

- The historical paired McNemar results are retained. The absence of new statistical testing must not be misrepresented as an absence of any previous significance test.

- Numbering is sequential: main Tables 1–12 followed by complete-record Tables 13–20; Figures 1–17. Every table and figure is referenced in body prose, including explicit Appendix C model mapping.

- The Phase 13 title, abstract, questions, method/metric explanations, direct RQ answers, reproducibility boundaries, and conclusion replace progress-report language. Future work is explicitly hypothetical, with no required experiment represented as unfinished.

## Unsupported claims excluded

No missing resource value, tokenizer revision, library version, exact prompt template, quantization outlier threshold, confidence interval, additional p-value, class support, raw confusion count, direct power reading, edge-device benchmark, lifecycle saving, or universal compression threshold is invented. No claim of exact event-disjoint or pretraining-contamination-free evaluation is made. No general superiority of DistilBERT over Qwen, statistically established rare-class trend, or repeated-seed ablation effect is asserted.

## Verification status

- PASS: existing Phase 12 DOCX used as the editable base; 20 table bodies and 17 image assets preserved exactly.

- PASS: all eight master records and additional Phase 11/12 metric fields retained, including collapsed and incomplete variants; source missingness preserved.

- PASS: 205-word abstract, eight keywords, explicit RQs and answers, methods/metrics/setup, Green AI implications, limits, reproducibility, conclusion and references included.

- PASS: DOCX and self-contained Markdown share ordered manuscript text, table data, and figure payloads. The Markdown embeds PNGs as data URIs; display depends on the reader supporting data-URI images. DOCX/PDF provide the fixed-layout reference.

- PASS: all 35 rendered pages inspected; final changed pages re-inspected. No blank pages, clipped page content, or orphan headings found. PDF contains eight reference hyperlinks. The PDF is rendered from the delivered DOCX.

- PASS: 175 Phase 11/12 source fields checked against the model-record tables with no discrepancies; the disclosed Pareto definition reproduces all five archived flags.

- LIMITATION: end-to-end experiment reproduction cannot pass with the supplied summary-only backups. The report explicitly limits verification to available evidence. Author and affiliation metadata were later supplied by Mayuresh Sharma; this administrative update does not alter the evidence assessment.

## References checked

Bibliographic links [1]–[8] were checked against the linked arXiv records, the HumAID DOI/publisher record, and official CodeCarbon documentation/site. Publication metadata is retained; a contemporary documentation page is not used to infer the experimental software configuration.
