# Solve rate by maximum state length reached: test_id

- source: `runs/E2-test_id`, budget 1000; max_state_tokens = longest [CLS] state [SEP] goal [SEP] among the states the search stored
- cutoff 126 tokens: the longest pair in training (RoPE length extrapolation check)
- **Descriptive, not causal:** max_state_tokens is partly an outcome of the search itself (longer searches reach longer states), so unsolved runs, which use the whole budget, tend to have longer states.

| method | max_state_tokens | n | solved | rate [95% CI] |
| --- | ---: | ---: | ---: | ---: |
| `neural-w1` | <= 126 | 300 | 300 | 1.000 [0.987, 1.000] |
| `neural-w1` | > 126 | 0 | 0 | – |
| `bestfirst-symbol_bag-w1` | <= 126 | 300 | 279 | 0.930 [0.895, 0.954] |
| `bestfirst-symbol_bag-w1` | > 126 | 0 | 0 | – |
