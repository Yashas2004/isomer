import pytest

from isomer.data.build import BuildConfig, build


def tiny_v2_config(out) -> BuildConfig:
    """A tiny IsoBench v2 config: every pool and split rule exercised in seconds."""
    return BuildConfig(out=str(out), train_per_theory=10, eval_per_theory=5, heldout_per_theory=3,
                       train_label_budget=300, label_budget=500, label_max_nodes=50_000, workers=1)


@pytest.fixture(scope="session")
def tiny_data(tmp_path_factory):
    """A tiny IsoBench build shared by the bench and report tests (~5 s, built once)."""
    out = tmp_path_factory.mktemp("isobench")
    build(tiny_v2_config(out), progress=False)
    return out


@pytest.fixture(scope="session")
def tiny_ckpt(tmp_path_factory):
    """A tiny randomly initialized distance model (needs torch), so CI covers the neural paths."""
    torch = pytest.importorskip("torch")
    from isomer.learn.model import DistanceModel, ModelConfig, save_checkpoint

    torch.manual_seed(0)
    model = DistanceModel(ModelConfig(d_model=32, n_layers=2, n_heads=4, ffn_hidden=86))
    path = tmp_path_factory.mktemp("ckpt") / "tiny.pt"
    save_checkpoint(path, model)
    return str(path)
