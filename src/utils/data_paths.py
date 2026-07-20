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


def _candidate_smap_paths(raw_root: str | Path | None):
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
        "data/SMAP_MSL",
        "external/tranad_upstream/data/SMAP_MSL",
        "external/OmniAnomaly/data",
    ):
        for item in _push(fallback):
            yield item


def _candidate_exathlon_paths(raw_root: str | Path | None):
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
        "data/exathlon_raw",
        "external/exathlon",
        "external/exathlon/data",
        "external/exathlon/data/raw",
    ):
        for item in _push(fallback):
            yield item


def _candidate_hai_paths(raw_root: str | Path | None):
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
        "data/hai_raw",
        "external/hai",
        "external/hai/hai-21.03",
    ):
        for item in _push(fallback):
            yield item


def _candidate_swat_paths(raw_root: str | Path | None):
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
        "data/SWaT",
        "external/SWaT",
        "external/tranad_upstream/data/SWaT",
    ):
        for item in _push(fallback):
            yield item


def _candidate_skab_paths(raw_root: str | Path | None):
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
        "data/skab_raw",
        "external/SKAB",
        "external/skab",
    ):
        for item in _push(fallback):
            yield item


def _candidate_wadi_paths(raw_root: str | Path | None):
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
        "data/WADI",
        "data/wadi_raw",
        "external/WADI",
        "external/tranad_upstream/data/WADI",
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


def has_raw_smap_layout(path: str | Path) -> bool:
    root = Path(path)
    return (
        (root / "train").exists()
        and (root / "test").exists()
        and (root / "labeled_anomalies.csv").exists()
    )


def has_raw_exathlon_layout(path: str | Path) -> bool:
    root = Path(path)
    if (root / "ground_truth.csv").exists() or (root / "ground_truth.zip").exists():
        return any(p.is_dir() and p.name.lower().startswith("app") for p in root.iterdir())
    data_dir = root / "data"
    raw_dir = data_dir / "raw"
    return ((data_dir / "ground_truth.csv").exists() or (raw_dir / "ground_truth.zip").exists()) and raw_dir.exists()


def _is_hai_version_dir(path: Path) -> bool:
    has_train = any(path.glob("train*.csv")) or any(path.glob("train*.csv.gz"))
    has_test = any(path.glob("test*.csv")) or any(path.glob("test*.csv.gz"))
    return has_train and has_test


def has_raw_hai_layout(path: str | Path) -> bool:
    root = Path(path)
    if _is_hai_version_dir(root):
        return True
    return any(child.is_dir() and _is_hai_version_dir(child) for child in root.glob("hai-*"))


def has_raw_swat_layout(path: str | Path) -> bool:
    root = Path(path)
    normal = root / "SWaT_Dataset_Normal_v1.csv"
    attack = root / "SWaT_Dataset_Attack_v0.csv"
    if normal.exists() and attack.exists():
        return True
    csvs = list(root.glob("*.csv"))
    return any("normal" in p.name.lower() for p in csvs) and any("attack" in p.name.lower() for p in csvs)


def _is_skab_data_dir(path: Path) -> bool:
    required = ("anomaly-free", "valve1", "valve2", "other")
    return all((path / name).exists() for name in required)


def has_raw_skab_layout(path: str | Path) -> bool:
    root = Path(path)
    if _is_skab_data_dir(root):
        return True
    data_dir = root / "data"
    return _is_skab_data_dir(data_dir)


def _has_wadi_train_file(root: Path) -> bool:
    return any(root.glob("WADI_14days*.csv"))


def _has_wadi_test_file(root: Path) -> bool:
    return any(root.glob("WADI_attackdata*.csv"))


def has_raw_wadi_layout(path: str | Path) -> bool:
    root = Path(path)
    if _has_wadi_train_file(root) and _has_wadi_test_file(root):
        return True
    data_dir = root / "data"
    return _has_wadi_train_file(data_dir) and _has_wadi_test_file(data_dir)


def resolve_raw_smd_root(raw_root: str | Path | None = None) -> Path:
    for candidate in _candidate_paths(raw_root):
        if candidate.exists() and has_raw_smd_layout(candidate):
            return candidate.resolve()
    if raw_root is not None:
        return Path(raw_root)
    return project_root() / "data" / "ServerMachineDataset"


def resolve_raw_smap_root(raw_root: str | Path | None = None) -> Path:
    for candidate in _candidate_smap_paths(raw_root):
        if candidate.exists() and has_raw_smap_layout(candidate):
            return candidate.resolve()
    if raw_root is not None:
        return Path(raw_root)
    return project_root() / "data" / "SMAP_MSL"


def resolve_raw_exathlon_root(raw_root: str | Path | None = None) -> Path:
    for candidate in _candidate_exathlon_paths(raw_root):
        if candidate.exists() and has_raw_exathlon_layout(candidate):
            return candidate.resolve()
    if raw_root is not None:
        return Path(raw_root)
    return project_root() / "external" / "exathlon"


def resolve_raw_hai_root(raw_root: str | Path | None = None, version: str = "21.03") -> Path:
    wanted_name = f"hai-{version}"
    for candidate in _candidate_hai_paths(raw_root):
        if not candidate.exists():
            continue
        if _is_hai_version_dir(candidate):
            if candidate.name == wanted_name or version is None:
                return candidate.resolve()
        wanted_dir = candidate / wanted_name
        if wanted_dir.exists() and _is_hai_version_dir(wanted_dir):
            return wanted_dir.resolve()
        if version is None and has_raw_hai_layout(candidate):
            for child in sorted(candidate.glob("hai-*")):
                if child.is_dir() and _is_hai_version_dir(child):
                    return child.resolve()
    if raw_root is not None:
        raw_path = Path(raw_root)
        if raw_path.name == wanted_name:
            return raw_path
        return raw_path / wanted_name
    return project_root() / "external" / "hai" / wanted_name


def resolve_raw_swat_root(raw_root: str | Path | None = None) -> Path:
    for candidate in _candidate_swat_paths(raw_root):
        if candidate.exists() and has_raw_swat_layout(candidate):
            return candidate.resolve()
    if raw_root is not None:
        return Path(raw_root)
    return project_root() / "data" / "SWaT"


def resolve_raw_skab_root(raw_root: str | Path | None = None) -> Path:
    for candidate in _candidate_skab_paths(raw_root):
        if candidate.exists() and has_raw_skab_layout(candidate):
            return candidate.resolve()
    if raw_root is not None:
        return Path(raw_root)
    return project_root() / "external" / "SKAB"


def resolve_raw_wadi_root(raw_root: str | Path | None = None) -> Path:
    for candidate in _candidate_wadi_paths(raw_root):
        if candidate.exists() and has_raw_wadi_layout(candidate):
            return candidate.resolve()
    if raw_root is not None:
        return Path(raw_root)
    return project_root() / "external" / "WADI"


def resolve_smd_label_dir(raw_root: str | Path) -> Path:
    root = resolve_raw_smd_root(raw_root)
    test_label = root / "test_label"
    if test_label.exists():
        return test_label
    labels = root / "labels"
    if labels.exists():
        return labels
    return test_label
