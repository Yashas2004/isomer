# E3 checkpoints

Every model in E3 has the h-v1 configuration: 6 layers, d=256, 4,731,649
parameters, Huber + listwise ranking loss, 20 epochs, bf16, batches of 64
problems. The models differ only in the training seed and the training-state
directory. h-v1 is the E2 checkpoint (release
[v0.2-e2](https://github.com/Yashas2004/isomer/releases/tag/v0.2-e2)). The
other six are in release
[v0.3-e3](https://github.com/Yashas2004/isomer/releases/tag/v0.3-e3), renamed
to `<model>.pt`, with `SHA256SUMS.txt`.

| model | training data | train problems (exact distance) | train states | seed | sha256 | download |
|---|---|---:|---:|---:|---|---|
| h-v1 | `data/states-v2` | 29,268 (<= 4) | 551,072 | 0 | `928015f6250a8531e6249e799507d4ca8bd433e50e28b7639aae6fa885376c00` | [best.pt](https://github.com/Yashas2004/isomer/releases/download/v0.2-e2/best.pt) (v0.2-e2) |
| h-v1-s1 | `data/states-v2` | 29,268 (<= 4) | 551,072 | 1 | `9982b0d574d663b88ab2524b576979f42f0e45ba9a090367c9b55aec45f61291` | [h-v1-s1.pt](https://github.com/Yashas2004/isomer/releases/download/v0.3-e3/h-v1-s1.pt) |
| h-v1-s2 | `data/states-v2` | 29,268 (<= 4) | 551,072 | 2 | `7f6864613e9c8bbf90b913f3c95b13c74c19236f6ab56c8af251650a55daef1d` | [h-v1-s2.pt](https://github.com/Yashas2004/isomer/releases/download/v0.3-e3/h-v1-s2.pt) |
| h-d6-s0 | `data/states-v2-d6` | 36,045 (<= 6) | 792,841 | 0 | `d5ca6f32a62c97a92bc486b2e418cce6f3c96a704d98b280a108c18805e36701` | [h-d6-s0.pt](https://github.com/Yashas2004/isomer/releases/download/v0.3-e3/h-d6-s0.pt) |
| h-d6-s1 | `data/states-v2-d6` | 36,045 (<= 6) | 792,841 | 1 | `e1dd6b3e29b236e85cae569454a6481a960c741de65a49eace843e9bdb81c51a` | [h-d6-s1.pt](https://github.com/Yashas2004/isomer/releases/download/v0.3-e3/h-d6-s1.pt) |
| h-d6-s2 | `data/states-v2-d6` | 36,045 (<= 6) | 792,841 | 2 | `f870d9057871fa3e151e61b37e48880e0c916bb78748af2be5197c7bf81714d5` | [h-d6-s2.pt](https://github.com/Yashas2004/isomer/releases/download/v0.3-e3/h-d6-s2.pt) |
| h-d6m-s0 | `data/states-v2-d6-matched` | 29,268 (<= 6) | 642,983 | 0 | `abffc4afd0d00e74b4b573e206e287b94416683a4f9f684bbe3d66aae4cece69` | [h-d6m-s0.pt](https://github.com/Yashas2004/isomer/releases/download/v0.3-e3/h-d6m-s0.pt) |

- **Training data.** `data/states-v2` comes from the IsoBench v2 train split. `data/states-v2-d6` comes from `data/isobench-v2-d6`, which has the same train pool relabelled with distances up to 6 (`scripts/e3_build_d6.py`). `data/states-v2-d6-matched` is a seeded random subset of the d6 train problems, with as many problems as v2 (`scripts/e3_build_matched.py`; see `d6_matched_prediction.md`).
- **States.** Training states can be one step deeper than the problems: distance <= 5 for v2 and <= 7 for d6.
- **Sources.** Problem and state counts are from `d6_matched_prediction.md`. Seeds and data directories are from each run's `config.json`. Each sha256 is that of `runs/<model>/best.pt`, which the scoring and bench runs also record.
- **Checking a download:**
  - Windows: `certutil -hashfile h-d6-s0.pt SHA256`
  - Linux/macOS: `sha256sum -c SHA256SUMS.txt`
- **Using a checkpoint:** put it at `runs/<model>/best.pt`. The E3 scripts and `neural-w1@runs/<model>/best.pt#bf16-fast` expect that path.
