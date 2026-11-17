"""The numbers in the documentation are the manuscript's numbers.

Every figure quoted in README.md is recomputed here from the per-seed files in
`results/tables/`, so a stale table in the prose fails the suite instead of
misreporting a result. The margins, the "15 of 16" claim and the five-seed
convention are all checked against the same data the manuscript's tables were
generated from.
"""
import json
import os
import re
import statistics as st

import pytest

#: The ten conventional activations the paper compares against.
CONVENTIONAL = ("relu", "leaky_relu", "gelu", "silu", "mish", "hardswish",
                "softplus", "elu", "selu", "tanh")
DATASETS = ("fmnist", "cifar10", "cifar100", "food101")
LABEL = {"relu": "ReLU", "leaky_relu": "LeakyReLU", "gelu": "GELU",
         "silu": "SiLU", "mish": "Mish", "hardswish": "HardSwish",
         "softplus": "Softplus", "elu": "ELU", "selu": "SELU", "tanh": "Tanh"}
#: README row label -> (file, section) holding that target's common-config runs
ROWS = (("ViT-d2", "vit"), ("Swin-d2", "swin"), ("LinViT-d2", "linvit"),
        ("MLP-Mixer", "mlpmixer"))


@pytest.fixture(scope="module")
def common(repo_root):
    """{target: {dataset: {activation: [per-seed accuracies]}}}, common config."""
    tables = os.path.join(repo_root, "results", "tables")
    with open(os.path.join(tables, "transfer_5seed_raw.json")) as f:
        raw = json.load(f)["acc"]
    with open(os.path.join(tables, "mixer_5seed_raw.json")) as f:
        mixer = json.load(f)["common"]
    out = {}
    for key, arch in (("vit", "vit"), ("swin", "swin"), ("linvit", "linvit")):
        out[arch] = {ds: {k.split("|")[-1]: v for k, v in raw.items()
                          if k.startswith(f"{key}|{ds}|")} for ds in DATASETS}
    out["mlpmixer"] = {ds: {k.split("|")[-1]: v for k, v in mixer.items()
                            if k.startswith(f"{ds}|")} for ds in DATASETS}
    return out


def _margin(cells):
    """(margin in pp, name of the best conventional activation)."""
    fact = st.mean(cells["fact_fixed"])
    best = max((a for a in cells if a in CONVENTIONAL),
               key=lambda a: st.mean(cells[a]))
    return fact - st.mean(cells[best]), best


def _readme_margins(repo_root):
    """Parse the README's margin table into {target: {dataset: (value, label)}}."""
    with open(os.path.join(repo_root, "README.md")) as f:
        text = f.read()
    cell = re.compile(r"([+\-−][\d.]+)\s*\(([A-Za-z]+)\)")
    found = {}
    for label, arch in ROWS:
        line = next((ln for ln in text.splitlines()
                     if ln.startswith(f"| {label}")), None)
        assert line, f"no margin row for {label} in README.md"
        cells = cell.findall(line)
        assert len(cells) == 4, f"{label}: expected 4 margins, got {cells}"
        found[arch] = {ds: (float(v.replace("−", "-")), name)
                       for ds, (v, name) in zip(DATASETS, cells)}
    return found


def test_readme_margins_match_the_per_seed_numbers(repo_root, common):
    stated = _readme_margins(repo_root)
    for _, arch in ROWS:
        for ds in DATASETS:
            value, name = stated[arch][ds]
            margin, best = _margin(common[arch][ds])
            assert name == LABEL[best], (
                f"{arch}/{ds}: README names {name} as the best conventional "
                f"activation, the data says {LABEL[best]}")
            assert round(margin, 2) == value, (
                f"{arch}/{ds}: README says {value:+.2f} pp, the data says "
                f"{margin:+.2f} pp")


def test_the_transferred_curve_wins_fifteen_of_sixteen(common):
    """The headline claim of the paper's Section 4, under the common config."""
    wins = [(arch, ds) for _, arch in ROWS for ds in DATASETS
            if _margin(common[arch][ds])[0] > 0]
    losses = [(arch, ds) for _, arch in ROWS for ds in DATASETS
              if _margin(common[arch][ds])[0] <= 0]
    assert len(wins) == 15, f"expected 15 wins, got {len(wins)}"
    assert losses == [("mlpmixer", "food101")], (
        f"the paper's single exception is MLP-Mixer/Food-101, got {losses}")


def test_every_reported_cell_has_five_seeds(common):
    for _, arch in ROWS:
        for ds in DATASETS:
            for act, runs in common[arch][ds].items():
                assert len(runs) == 5, f"{arch}/{ds}/{act} has {len(runs)} seeds"


def test_readme_parameter_counts_match_the_built_models(repo_root, artifact_path):
    """The parameter ranges in the README are what the code actually builds."""
    from left.activations import FAct
    from left.models.linvit import build_linvit
    from left.models.mlp_mixer import build_mlpmixer
    from left.models.swin import build_swin
    from left.models.vit import build_vit

    builders = {"ViT": build_vit, "Swin": build_swin, "LinViT": build_linvit,
                "MLP-Mixer": lambda ds, act: build_mlpmixer(ds, act, embed_dim=76)}
    with open(os.path.join(repo_root, "README.md")) as f:
        text = f.read()
    for label, build in builders.items():
        counts = []
        for ds in DATASETS:
            model = build(ds, FAct.from_artifact(artifact_path))
            counts.append(sum(p.numel() for p in model.parameters() if p.requires_grad))
        stated = re.search(rf"\*\*{re.escape(label)}\*\*.*?\|\s*([\d,]+)\s*–\s*([\d,]+)\s*\|",
                           text)
        assert stated, f"no parameter range for {label} in README.md"
        low, high = (int(g.replace(",", "")) for g in stated.groups())
        assert (low, high) == (min(counts), max(counts)), (
            f"{label}: README says {low:,}-{high:,}, the models give "
            f"{min(counts):,}-{max(counts):,}")


def test_readme_quotes_the_artifact_coefficients(repo_root, artifact_path):
    with open(artifact_path) as f:
        art = json.load(f)
    theta = [art["a0"], art["a"][0], art["a"][1], art["b"][0], art["b"][1]]
    with open(os.path.join(repo_root, "README.md")) as f:
        text = f.read()
    for value in theta:
        rounded = f"{value:.4f}".replace("-", "−")
        assert rounded in text, f"README does not quote the coefficient {rounded}"
