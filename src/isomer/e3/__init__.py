"""E3: diagnostics of the learned heuristic (calibration, ranking, invariance).

Nothing here changes the kernel, the rewrite semantics, the problem pool or
the benchmark splits. Modules:

* ``regen``      regenerate a published stratified sample without the full build
* ``children``   exact optimal/non-optimal labels for every legal child of a state
* ``rename``     meaning-preserving renamings of problems (skolems, theory symbols)
"""
