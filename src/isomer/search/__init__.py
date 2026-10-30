"""Search algorithms. Importing the package registers every searcher in
``common.SEARCHERS`` (and the path-parameterized ones in ``common.PARAMETRIC``).
``neural`` imports torch only when a model is loaded."""

from . import best_first, bfs, neural  # noqa: F401
