# Solve rate by maximum state length reached: test_deep

- source: `runs/E2-test_deep`, budget 1000; max_state_tokens = longest [CLS] state [SEP] goal [SEP] among the states the search stored
- cutoff 126 tokens: the longest pair in training (RoPE length extrapolation check)
- **Descriptive, not causal:** max_state_tokens is partly an outcome of the search itself (longer searches reach longer states), so unsolved runs, which use the whole budget, tend to have longer states.

| method | max_state_tokens | n | solved | rate [95% CI] |
| --- | ---: | ---: | ---: | ---: |
| `neural-w1` | <= 126 | 68 | 2 | 0.029 [0.008, 0.101] |
| `neural-w1` | > 126 | 32 | 0 | 0.000 [0.000, 0.107] |
| `bestfirst-symbol_bag-w1` | <= 126 | 67 | 0 | 0.000 [0.000, 0.054] |
| `bestfirst-symbol_bag-w1` | > 126 | 33 | 0 | 0.000 [0.000, 0.104] |
