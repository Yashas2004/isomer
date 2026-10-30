# Solve rate by maximum state length reached: test_theory

- source: `runs/E2-test_theory`, budget 1000; max_state_tokens = longest [CLS] state [SEP] goal [SEP] among the states the search stored
- cutoff 126 tokens: the longest pair in training (RoPE length extrapolation check)
- **Descriptive, not causal:** max_state_tokens is partly an outcome of the search itself (longer searches reach longer states), so unsolved runs, which use the whole budget, tend to have longer states.

| method | max_state_tokens | n | solved | rate [95% CI] |
| --- | ---: | ---: | ---: | ---: |
| `neural-w1` | <= 126 | 297 | 169 | 0.569 [0.512, 0.624] |
| `neural-w1` | > 126 | 3 | 0 | 0.000 [0.000, 0.561] |
| `bestfirst-symbol_bag-w1` | <= 126 | 297 | 166 | 0.559 [0.502, 0.614] |
| `bestfirst-symbol_bag-w1` | > 126 | 3 | 0 | 0.000 [0.000, 0.561] |
