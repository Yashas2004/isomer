# E3: overlap between evaluated states and training states

A pair is (theory, state, goal) with the exact strings the tokenizer sees. "In v2" / "in d6" means the pair is a line of `train.jsonl` in `data/states-v2` / `data/states-v2-d6`. Problem ids of the test splits are disjoint from the train pool; this measures the (state, goal) pairs, which the build does not keep disjoint.

| training set | directory | distinct training pairs |
|---:|---:|---:|
| v2 | `data/states-v2` | 544910 |
| d6 | `data/states-v2-d6` | 786051 |

## 1. Overlap of scored states and of their children

"State in X": the on-path state's own pair is a training pair. "Children in X": share of all (child, goal) pairs that are training pairs. "States with a child in X": at least one child.

### test_depth, by distance of the state

| k | scored states | state in v2 | state in d6 | children | children in v2 | children in d6 | states with a child in v2 | states with a child in d6 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 300 | 1 (0.3%) | 1 (0.3%) | 14038 | 8 (0.1%) | 8 (0.1%) | 1 (0.3%) | 1 (0.3%) |
| 2 | 300 | 1 (0.3%) | 1 (0.3%) | 12619 | 9 (0.1%) | 9 (0.1%) | 1 (0.3%) | 1 (0.3%) |
| 3 | 300 | 1 (0.3%) | 1 (0.3%) | 10909 | 8 (0.1%) | 8 (0.1%) | 1 (0.3%) | 1 (0.3%) |
| 4 | 300 | 1 (0.3%) | 1 (0.3%) | 8928 | 6 (0.1%) | 6 (0.1%) | 1 (0.3%) | 1 (0.3%) |
| 5 | 300 | 1 (0.3%) | 1 (0.3%) | 6907 | 1 (0.0%) | 1 (0.0%) | 1 (0.3%) | 1 (0.3%) |
| 6 | 230 | 0 (0.0%) | 0 (0.0%) | 4335 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 7 | 124 | 0 (0.0%) | 0 (0.0%) | 1558 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 8 | 64 | 0 (0.0%) | 0 (0.0%) | 497 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 9 | 15 | 0 (0.0%) | 0 (0.0%) | 54 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 10 | 2 | 0 (0.0%) | 0 (0.0%) | 2 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| all | 1935 | 5 (0.3%) | 5 (0.3%) | 59847 | 32 (0.1%) | 32 (0.1%) | 5 (0.3%) | 5 (0.3%) |

- v2: of the 5 scored states whose pair is in training, 5 carry the same distance label there as in the test labels; 0 are in `val.jsonl` (used to pick the checkpoint).
- d6: of the 5 scored states whose pair is in training, 5 carry the same distance label there as in the test labels; 0 are in `val.jsonl` (used to pick the checkpoint).

### test_id, by distance of the state

| k | scored states | state in v2 | state in d6 | children | children in v2 | children in d6 | states with a child in v2 | states with a child in d6 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 300 | 16 (5.3%) | 17 (5.7%) | 8061 | 121 (1.5%) | 129 (1.6%) | 25 (8.3%) | 26 (8.7%) |
| 2 | 248 | 12 (4.8%) | 13 (5.2%) | 5874 | 56 (1.0%) | 64 (1.1%) | 17 (6.9%) | 18 (7.3%) |
| 3 | 171 | 5 (2.9%) | 6 (3.5%) | 3518 | 14 (0.4%) | 21 (0.6%) | 8 (4.7%) | 9 (5.3%) |
| 4 | 85 | 1 (1.2%) | 2 (2.4%) | 1454 | 4 (0.3%) | 11 (0.8%) | 2 (2.4%) | 3 (3.5%) |
| all | 804 | 34 (4.2%) | 38 (4.7%) | 18907 | 195 (1.0%) | 225 (1.2%) | 52 (6.5%) | 56 (7.0%) |

- v2: of the 34 scored states whose pair is in training, 34 carry the same distance label there as in the test labels; 5 are in `val.jsonl` (used to pick the checkpoint).
- d6: of the 38 scored states whose pair is in training, 38 carry the same distance label there as in the test labels; 5 are in `val.jsonl` (used to pick the checkpoint).

## 2. All-children top-1 on non-overlapping states

Overlap is taken against each model's own training set. "State unseen": the state's pair is not a training pair. "State and children unseen": in addition no child's pair is. Top-1 is over all children as before (ties split); only the set of states changes, no score is recomputed.

### test_depth, all distances (95% CI: cluster bootstrap over problems)

| model | trained on | all states | top-1 | state unseen | top-1 | state and children unseen | top-1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| h-v1 | v2 | 1935 | 0.719 (0.690 to 0.747) | 1930 | 0.718 (0.692 to 0.745) | 1930 | 0.718 (0.692 to 0.745) |
| h-v1-s1 | v2 | 1935 | 0.800 (0.773 to 0.826) | 1930 | 0.799 (0.773 to 0.824) | 1930 | 0.799 (0.773 to 0.824) |
| h-v1-s2 | v2 | 1935 | 0.645 (0.615 to 0.675) | 1930 | 0.644 (0.613 to 0.676) | 1930 | 0.644 (0.613 to 0.676) |
| h-d6-s0 | d6 | 1935 | 0.953 (0.941 to 0.964) | 1930 | 0.953 (0.941 to 0.964) | 1930 | 0.953 (0.941 to 0.964) |

#### test_depth, h-v1, by distance of the state

| k | all states | top-1 | state unseen | top-1 | state and children unseen | top-1 | symbol_bag on the last subset |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 300 | 0.823 | 299 | 0.823 | 299 | 0.823 | 0.160 |
| 2 | 300 | 0.730 | 299 | 0.729 | 299 | 0.729 | 0.190 |
| 3 | 300 | 0.740 | 299 | 0.739 | 299 | 0.739 | 0.157 |
| 4 | 300 | 0.690 | 299 | 0.689 | 299 | 0.689 | 0.163 |
| 5 | 300 | 0.653 | 299 | 0.652 | 299 | 0.652 | 0.230 |
| 6 | 230 | 0.657 | 230 | 0.657 | 230 | 0.657 | 0.253 |
| 7 | 124 | 0.694 | 124 | 0.694 | 124 | 0.694 | 0.332 |
| 8 | 64 | 0.734 | 64 | 0.734 | 64 | 0.734 | 0.368 |
| 9 | 15 | 0.933 | 15 | 0.933 | 15 | 0.933 | 0.563 |
| 10 | 2 | 1.000 | 2 | 1.000 | 2 | 1.000 | 1.000 |

#### test_depth, h-v1-s1, by distance of the state

| k | all states | top-1 | state unseen | top-1 | state and children unseen | top-1 | symbol_bag on the last subset |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 300 | 0.910 | 299 | 0.910 | 299 | 0.910 | 0.160 |
| 2 | 300 | 0.847 | 299 | 0.846 | 299 | 0.846 | 0.190 |
| 3 | 300 | 0.837 | 299 | 0.836 | 299 | 0.836 | 0.157 |
| 4 | 300 | 0.813 | 299 | 0.813 | 299 | 0.813 | 0.163 |
| 5 | 300 | 0.730 | 299 | 0.729 | 299 | 0.729 | 0.230 |
| 6 | 230 | 0.687 | 230 | 0.687 | 230 | 0.687 | 0.253 |
| 7 | 124 | 0.685 | 124 | 0.685 | 124 | 0.685 | 0.332 |
| 8 | 64 | 0.750 | 64 | 0.750 | 64 | 0.750 | 0.368 |
| 9 | 15 | 0.933 | 15 | 0.933 | 15 | 0.933 | 0.563 |
| 10 | 2 | 1.000 | 2 | 1.000 | 2 | 1.000 | 1.000 |

#### test_depth, h-v1-s2, by distance of the state

| k | all states | top-1 | state unseen | top-1 | state and children unseen | top-1 | symbol_bag on the last subset |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 300 | 0.727 | 299 | 0.726 | 299 | 0.726 | 0.160 |
| 2 | 300 | 0.667 | 299 | 0.666 | 299 | 0.666 | 0.190 |
| 3 | 300 | 0.613 | 299 | 0.612 | 299 | 0.612 | 0.157 |
| 4 | 300 | 0.640 | 299 | 0.639 | 299 | 0.639 | 0.163 |
| 5 | 300 | 0.617 | 299 | 0.615 | 299 | 0.615 | 0.230 |
| 6 | 230 | 0.583 | 230 | 0.583 | 230 | 0.583 | 0.253 |
| 7 | 124 | 0.629 | 124 | 0.629 | 124 | 0.629 | 0.332 |
| 8 | 64 | 0.672 | 64 | 0.672 | 64 | 0.672 | 0.368 |
| 9 | 15 | 0.800 | 15 | 0.800 | 15 | 0.800 | 0.563 |
| 10 | 2 | 1.000 | 2 | 1.000 | 2 | 1.000 | 1.000 |

#### test_depth, h-d6-s0, by distance of the state

| k | all states | top-1 | state unseen | top-1 | state and children unseen | top-1 | symbol_bag on the last subset |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 300 | 0.993 | 299 | 0.993 | 299 | 0.993 | 0.160 |
| 2 | 300 | 0.970 | 299 | 0.970 | 299 | 0.970 | 0.190 |
| 3 | 300 | 0.970 | 299 | 0.970 | 299 | 0.970 | 0.157 |
| 4 | 300 | 0.953 | 299 | 0.953 | 299 | 0.953 | 0.163 |
| 5 | 300 | 0.920 | 299 | 0.920 | 299 | 0.920 | 0.230 |
| 6 | 230 | 0.917 | 230 | 0.917 | 230 | 0.917 | 0.253 |
| 7 | 124 | 0.903 | 124 | 0.903 | 124 | 0.903 | 0.332 |
| 8 | 64 | 0.969 | 64 | 0.969 | 64 | 0.969 | 0.368 |
| 9 | 15 | 1.000 | 15 | 1.000 | 15 | 1.000 | 0.563 |
| 10 | 2 | 1.000 | 2 | 1.000 | 2 | 1.000 | 1.000 |

### test_id, all distances (95% CI: cluster bootstrap over problems)

| model | trained on | all states | top-1 | state unseen | top-1 | state and children unseen | top-1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| h-v1 | v2 | 804 | 0.908 (0.883 to 0.931) | 770 | 0.904 (0.878 to 0.928) | 752 | 0.902 (0.875 to 0.926) |
| h-v1-s1 | v2 | 804 | 0.968 (0.953 to 0.980) | 770 | 0.968 (0.952 to 0.980) | 752 | 0.968 (0.953 to 0.982) |
| h-v1-s2 | v2 | 804 | 0.912 (0.888 to 0.933) | 770 | 0.908 (0.883 to 0.929) | 752 | 0.908 (0.885 to 0.931) |
| h-d6-s0 | d6 | 804 | 0.991 (0.984 to 0.997) | 766 | 0.991 (0.984 to 0.997) | 748 | 0.991 (0.983 to 0.997) |

#### test_id, h-v1, by distance of the state

| k | all states | top-1 | state unseen | top-1 | state and children unseen | top-1 | symbol_bag on the last subset |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 300 | 0.973 | 284 | 0.972 | 275 | 0.971 | 0.258 |
| 2 | 248 | 0.903 | 236 | 0.898 | 231 | 0.896 | 0.307 |
| 3 | 171 | 0.871 | 166 | 0.867 | 163 | 0.865 | 0.307 |
| 4 | 85 | 0.765 | 84 | 0.762 | 83 | 0.759 | 0.330 |

#### test_id, h-v1-s1, by distance of the state

| k | all states | top-1 | state unseen | top-1 | state and children unseen | top-1 | symbol_bag on the last subset |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 300 | 1.000 | 284 | 1.000 | 275 | 1.000 | 0.258 |
| 2 | 248 | 0.960 | 236 | 0.958 | 231 | 0.957 | 0.307 |
| 3 | 171 | 0.953 | 166 | 0.952 | 163 | 0.951 | 0.307 |
| 4 | 85 | 0.906 | 84 | 0.917 | 83 | 0.928 | 0.330 |

#### test_id, h-v1-s2, by distance of the state

| k | all states | top-1 | state unseen | top-1 | state and children unseen | top-1 | symbol_bag on the last subset |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 300 | 0.980 | 284 | 0.979 | 275 | 0.978 | 0.258 |
| 2 | 248 | 0.899 | 236 | 0.894 | 231 | 0.896 | 0.307 |
| 3 | 171 | 0.848 | 166 | 0.843 | 163 | 0.840 | 0.307 |
| 4 | 85 | 0.835 | 84 | 0.833 | 83 | 0.843 | 0.330 |

#### test_id, h-d6-s0, by distance of the state

| k | all states | top-1 | state unseen | top-1 | state and children unseen | top-1 | symbol_bag on the last subset |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 300 | 1.000 | 283 | 1.000 | 274 | 1.000 | 0.259 |
| 2 | 248 | 0.996 | 235 | 0.996 | 230 | 0.996 | 0.307 |
| 3 | 171 | 0.977 | 165 | 0.976 | 162 | 0.975 | 0.308 |
| 4 | 85 | 0.976 | 83 | 0.976 | 82 | 0.976 | 0.328 |

### Mean prediction and drop to the optimal child on non-overlapping test_depth states

The quantities of the d6 prediction, on the strictest subset (state and children unseen), next to the value on all scored states.

| k | model | all states | mean prediction | drop | state and children unseen | mean prediction | drop |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | h-v1 | 300 | 4.238 | 0.590 | 299 | 4.237 | 0.590 |
| 5 | h-v1-s1 | 300 | 4.268 | 0.566 | 299 | 4.267 | 0.566 |
| 5 | h-v1-s2 | 300 | 4.201 | 0.604 | 299 | 4.200 | 0.604 |
| 5 | h-d6-s0 | 300 | 4.798 | 0.992 | 299 | 4.798 | 0.992 |
| 6 | h-v1 | 230 | 4.559 | 0.303 | 230 | 4.559 | 0.303 |
| 6 | h-v1-s1 | 230 | 4.542 | 0.295 | 230 | 4.542 | 0.295 |
| 6 | h-v1-s2 | 230 | 4.551 | 0.344 | 230 | 4.551 | 0.344 |
| 6 | h-d6-s0 | 230 | 5.617 | 0.798 | 230 | 5.617 | 0.798 |
| 7 | h-v1 | 124 | 4.688 | 0.068 | 124 | 4.688 | 0.068 |
| 7 | h-v1-s1 | 124 | 4.694 | 0.119 | 124 | 4.694 | 0.119 |
| 7 | h-v1-s2 | 124 | 4.685 | 0.139 | 124 | 4.685 | 0.139 |
| 7 | h-d6-s0 | 124 | 6.234 | 0.578 | 124 | 6.234 | 0.578 |
| 8 | h-v1 | 64 | 4.764 | 0.076 | 64 | 4.764 | 0.076 |
| 8 | h-v1-s1 | 64 | 4.703 | 0.072 | 64 | 4.703 | 0.072 |
| 8 | h-v1-s2 | 64 | 4.708 | 0.056 | 64 | 4.708 | 0.056 |
| 8 | h-d6-s0 | 64 | 6.646 | 0.506 | 64 | 6.646 | 0.506 |

## 3. Search on test_depth problems

Per problem: is the start pair (lhs, rhs) a training pair, and is any (state, goal) pair on its stored shortest proof (distance >= 1, including states the labeller left unknown) a training pair.

| problem distance | problems | start pair in v2 | any on-path pair in v2 | start pair in d6 | any on-path pair in d6 |
|---:|---:|---:|---:|---:|---:|
| 5 | 70 | 1 (1.4%) | 1 (1.4%) | 1 (1.4%) | 1 (1.4%) |
| 6 | 97 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 7 | 58 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 8 | 52 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 9 | 19 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 10 | 4 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| all | 300 | 1 (0.3%) | 1 (0.3%) | 1 (0.3%) | 1 (0.3%) |

Solved (kernel-accepted) by neural-w1, on all problems and on the two parts:

| model | trained on | problems | n | solved @30 | solved @100 | solved @300 | solved @1000 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| h-v1 | v2 | all problems | 300 | 139 (46.3%) | 173 (57.7%) | 202 (67.3%) | 224 (74.7%) |
| h-v1 | v2 | no on-path pair in training | 299 | 138 (46.2%) | 172 (57.5%) | 201 (67.2%) | 223 (74.6%) |
| h-v1 | v2 | some on-path pair in training | 1 | 1 (100.0%) | 1 (100.0%) | 1 (100.0%) | 1 (100.0%) |
| h-v1-s1 | v2 | all problems | 300 | 180 (60.0%) | 202 (67.3%) | 222 (74.0%) | 240 (80.0%) |
| h-v1-s1 | v2 | no on-path pair in training | 299 | 179 (59.9%) | 201 (67.2%) | 221 (73.9%) | 239 (79.9%) |
| h-v1-s1 | v2 | some on-path pair in training | 1 | 1 (100.0%) | 1 (100.0%) | 1 (100.0%) | 1 (100.0%) |
| h-v1-s2 | v2 | all problems | 300 | 124 (41.3%) | 152 (50.7%) | 176 (58.7%) | 195 (65.0%) |
| h-v1-s2 | v2 | no on-path pair in training | 299 | 123 (41.1%) | 151 (50.5%) | 175 (58.5%) | 194 (64.9%) |
| h-v1-s2 | v2 | some on-path pair in training | 1 | 1 (100.0%) | 1 (100.0%) | 1 (100.0%) | 1 (100.0%) |
| h-d6-s0 | d6 | all problems | 300 | 265 (88.3%) | 287 (95.7%) | 292 (97.3%) | 298 (99.3%) |
| h-d6-s0 | d6 | no on-path pair in training | 299 | 264 (88.3%) | 286 (95.7%) | 291 (97.3%) | 297 (99.3%) |
| h-d6-s0 | d6 | some on-path pair in training | 1 | 1 (100.0%) | 1 (100.0%) | 1 (100.0%) | 1 (100.0%) |

## Limits

- Only exact pairs are counted. A pair that differs by a renaming, by the direction of the equation (the goal as state), or by one rewrite is not counted as overlap.
- Search visits many states that are not on the stored proof; their overlap with training is not measured here. "No on-path pair in training" does not mean the search saw no training pair.
- The non-overlapping states are a different population from all scored states (they may be harder or easier), so a change in top-1 between the columns is not only the effect of recall. symbol_bag on the same subset is given for reference.
- Unknown states (labelling node cap) are excluded from sections 1 and 2, as everywhere in E3.
