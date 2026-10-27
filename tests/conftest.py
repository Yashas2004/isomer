import pytest

from isomer.data.build import BuildConfig, build


@pytest.fixture(scope="session")
def tiny_data(tmp_path_factory):
    """A tiny IsoBench build shared by the bench and report tests (~5 s, built once)."""
    out = tmp_path_factory.mktemp("isobench")
    build(BuildConfig(out=str(out), per_theory=10, per_heldout_theory=3, bfs_budget=300, workers=1),
          progress=False)
    return out
