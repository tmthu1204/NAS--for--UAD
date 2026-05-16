from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _candidate_paths(raw_root: str | Path | None):
    proj = project_root()
    seen: set[str] = set()

    def _push(path_like):
        if path_like is None:
            return
        p = Path(path_like)
        variants = [p] if p.is_absolute() else [proj / p, p]
        for item in variants:
            try:
                key = str(item.resolve(strict=False))
            except Exception:
                key = str(item)
            if key in seen:
                continue
            seen.add(key)
            yield item

    if raw_root is not None:
        for item in _push(raw_root):
            yield item

    for fallback in (
        "data/ServerMachineDataset",
        "external/OmniAnomaly/ServerMachineDataset",
        "external/tranad_upstream/data/SMD",
    ):
        for item in _push(fallback):
            yield item


def has_raw_smd_layout(path: str | Path) -> bool:
    root = Path(path)
    return (
        (root / "train").exists()
        and (root / "test").exists()
        and ((root / "test_label").exists() or (root / "labels").exists())
    )


def resolve_raw_smd_root(raw_root: str | Path | None = None) -> Path:
    for candidate in _candidate_paths(raw_root):
        if candidate.exists() and has_raw_smd_layout(candidate):
            return candidate.resolve()
    if raw_root is not None:
        return Path(raw_root)
    return project_root() / "data" / "ServerMachineDataset"


def resolve_smd_label_dir(raw_root: str | Path) -> Path:
    root = resolve_raw_smd_root(raw_root)
    test_label = root / "test_label"
    if test_label.exists():
        return test_label
    labels = root / "labels"
    if labels.exists():
        return labels
    return test_label
