# Solve rate by maximum state length reached: test_depth

- source: `runs/E2-test_depth`, budget 1000; max_state_tokens = longest [CLS] state [SEP] goal [SEP] among the states the search stored
- cutoff 126 tokens: the longest pair in training (RoPE length extrapolation check)
- **Descriptive, not causal:** max_state_tokens is partly an outcome of the search itself (longer searches reach longer states), so unsolved runs, which use the whole budget, tend to have longer states.

| method | max_state_tokens | n | solved | rate [95% CI] |
| --- | ---: | ---: | ---: | ---: |
| `neural-w1` | <= 126 | 294 | 224 | 0.762 [0.710, 0.807] |
| `neural-w1` | > 126 | 6 | 0 | 0.000 [0.000, 0.390] |
| `bestfirst-symbol_bag-w1` | <= 126 | 294 | 60 | 0.204 [0.162, 0.254] |
| `bestfirst-symbol_bag-w1` | > 126 | 6 | 0 | 0.000 [0.000, 0.390] |
