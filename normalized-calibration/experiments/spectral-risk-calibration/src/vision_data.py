"""Vision object hypergraphs: ID-explicit loading, frozen split, feature noise.

Release layout (ED-HNN preprocessed, HGNN-style text format)
------------------------------------------------------------
``.content`` rows: ``id  f_1 ... f_100  label``.  The first ``n_objects`` IDs
are the classification objects.  The remaining rows are auxiliary
hyperedge-node representations (label 0, not objects, never classified).

``.edges`` rows: ``object_id  hyperedge_id``.  The first published column is
the OBJECT and the second is the hyperedge; hyperedge IDs occupy
``[n_objects, n_objects + n_edges)``.  This orientation is verified, not
assumed: every hyperedge must contain exactly five distinct object IDs.

The published slot order is preserved.  Edges are canonicalised only by
sorting the five members inside a slot (an unordered set), never by changing
slots, dropping edges or filtering by labels.

All object features are available (transductive setting) as released.  The
train/validation/test masks are fixed once by a stratified 60/20/20 split;
features are standardised with TRAIN-NODE statistics only.  Test labels are
never passed to the predictor or the monitor.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]

DATASET_SPECS = {
    "ntu2012": {
        "content": "data/vision-hypergraphs/NTU2012.content",
        "edges": "data/vision-hypergraphs/NTU2012.edges",
        "content_sha256":
            "f59d6f62c94d1ef4861a95ddcd76c24411d7deb6fd4f2777e40ea83563630d2c",
        "edges_sha256":
            "3de34493d0e37b2d1b7d5df35e0d9246e4d3dff0de39ec4fa7f16abdfacee8b8",
        "n_objects": 2012,
        "n_classes": 67,
        "label_offset": 0,
        "feature_dim": 100,
        "edge_size": 5,
        "dataset_index": 1,
    },
    "modelnet40": {
        "content": "data/vision-hypergraphs/ModelNet40.content",
        "edges": "data/vision-hypergraphs/ModelNet40.edges",
        "content_sha256":
            "51565c050db09532ea16457bb04958aa5dec8acaab1f9009df47771ce76b8604",
        "edges_sha256":
            "e6e0ae37a39369b4a55a230554148e0187c58ee47f6e650b62f5436c6eac4048",
        "n_objects": 12311,
        "n_classes": 40,
        "label_offset": 1,
        "feature_dim": 100,
        "edge_size": 5,
        "dataset_index": 2,
    },
}


class VisionDataError(ValueError):
    """Raised when the released files violate the audited layout."""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class VisionData:
    name: str
    X: np.ndarray              # (n, d) raw object features, object-ID order
    labels: np.ndarray         # (n,) int64, remapped to 0..C-1
    edges: tuple               # m slots, each a sorted tuple of 5 object IDs
    object_rows: np.ndarray    # content row index for each object ID
    n_extra_rows: int
    content_sha256: str
    edges_sha256: str
    n_classes: int
    feature_dim: int


def _parse_int(tok, what):
    try:
        val = float(tok)
    except ValueError as exc:
        raise VisionDataError(f"{what}: not numeric: {tok!r}") from exc
    ival = int(round(val))
    if abs(val - ival) > 1e-9:
        raise VisionDataError(f"{what}: non-integral ID {tok!r}")
    return ival


def parse_vision_files(content_path, edges_path, n_objects, n_classes,
                       label_offset, feature_dim, edge_size=5,
                       source_name="vision"):
    """Parse and validate one released pair; explicit counts, no inference."""
    content_path, edges_path = Path(content_path), Path(edges_path)
    ids, feats, labs = [], [], []
    with open(content_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != feature_dim + 2:
                raise VisionDataError(
                    f"{source_name}.content:{lineno}: expected "
                    f"{feature_dim + 2} fields, got {len(parts)}")
            ids.append(_parse_int(parts[0], f"{source_name}.content:{lineno} ID"))
            try:
                feats.append([float(v) for v in parts[1:-1]])
                labs.append(_parse_int(parts[-1],
                                       f"{source_name}.content:{lineno} label"))
            except ValueError as exc:
                raise VisionDataError(
                    f"{source_name}.content:{lineno}: non-numeric value") from exc
    ids = np.asarray(ids, dtype=np.int64)
    X_all = np.asarray(feats, dtype=np.float64)
    y_all = np.asarray(labs, dtype=np.int64)
    if len(ids) == 0:
        raise VisionDataError(f"{source_name}: empty content file")
    if len(np.unique(ids)) != len(ids):
        raise VisionDataError(f"{source_name}: duplicate content IDs")
    if not np.isfinite(X_all).all():
        raise VisionDataError(f"{source_name}: non-finite features")

    # Explicit object rows: IDs 0..n_objects-1 must all exist exactly once.
    id_to_row = {int(v): i for i, v in enumerate(ids)}
    missing = [u for u in range(n_objects) if u not in id_to_row]
    if missing:
        raise VisionDataError(
            f"{source_name}: missing object IDs, first {missing[:5]}")
    object_rows = np.asarray([id_to_row[u] for u in range(n_objects)],
                             dtype=np.int64)
    X = X_all[object_rows]
    raw_labels = y_all[object_rows]
    if label_offset:
        raw_labels = raw_labels - int(label_offset)
    if raw_labels.min() < 0 or raw_labels.max() >= n_classes:
        raise VisionDataError(
            f"{source_name}: remapped labels outside 0..{n_classes - 1}")
    if len(np.unique(raw_labels)) != n_classes:
        raise VisionDataError(
            f"{source_name}: expected {n_classes} classes, found "
            f"{len(np.unique(raw_labels))}")

    # Incidence: object_id, hyperedge_id; hyperedge IDs live above n_objects.
    by_edge = {}
    seen_pairs = set()
    with open(edges_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 2:
                raise VisionDataError(
                    f"{source_name}.edges:{lineno}: expected 2 fields")
            u = _parse_int(parts[0], f"{source_name}.edges:{lineno} object")
            e = _parse_int(parts[1], f"{source_name}.edges:{lineno} hyperedge")
            if not (0 <= u < n_objects):
                raise VisionDataError(
                    f"{source_name}.edges:{lineno}: object {u} outside "
                    f"0..{n_objects - 1}")
            if not (n_objects <= e < n_objects + n_objects):
                raise VisionDataError(
                    f"{source_name}.edges:{lineno}: hyperedge {e} outside "
                    f"published hyperedge-node range")
            if (u, e) in seen_pairs:
                raise VisionDataError(
                    f"{source_name}.edges:{lineno}: duplicate incidence")
            seen_pairs.add((u, e))
            by_edge.setdefault(e, []).append(u)

    expected_edges = list(range(n_objects, 2 * n_objects))
    if sorted(by_edge) != expected_edges:
        raise VisionDataError(
            f"{source_name}: hyperedge IDs are not exactly "
            f"{n_objects}..{2 * n_objects - 1}")
    edges = []
    for e in expected_edges:
        members = tuple(sorted(by_edge[e]))
        if len(members) != edge_size:
            raise VisionDataError(
                f"{source_name}: hyperedge {e} has {len(members)} objects, "
                f"expected {edge_size}")
        if len(set(members)) != len(members):
            raise VisionDataError(
                f"{source_name}: duplicate object inside hyperedge {e}")
        edges.append(members)
    return VisionData(
        name=source_name,
        X=X,
        labels=raw_labels.astype(np.int64),
        edges=tuple(edges),
        object_rows=object_rows,
        n_extra_rows=int(len(ids) - n_objects),
        content_sha256=sha256_file(content_path),
        edges_sha256=sha256_file(edges_path),
        n_classes=int(n_classes),
        feature_dim=int(feature_dim),
    )


def load_vision_dataset(name, root=ROOT, verify=True, spec=None):
    """Load a named released dataset after SHA-256 verification."""
    spec = dict(spec or DATASET_SPECS[name])
    content = Path(root) / spec["content"]
    edges = Path(root) / spec["edges"]
    if not content.exists() or not edges.exists():
        raise VisionDataError(f"{name}: missing data files under {root}")
    if verify:
        got_c = sha256_file(content)
        got_e = sha256_file(edges)
        if got_c != spec["content_sha256"]:
            raise VisionDataError(
                f"{name}: content hash mismatch {got_c} != "
                f"{spec['content_sha256']}")
        if got_e != spec["edges_sha256"]:
            raise VisionDataError(
                f"{name}: edges hash mismatch {got_e} != "
                f"{spec['edges_sha256']}")
    data = parse_vision_files(
        content, edges,
        n_objects=spec["n_objects"], n_classes=spec["n_classes"],
        label_offset=spec["label_offset"], feature_dim=spec["feature_dim"],
        edge_size=spec.get("edge_size", 5), source_name=name)
    if data.content_sha256 != spec["content_sha256"]:
        raise VisionDataError(f"{name}: content changed during read")
    if data.edges_sha256 != spec["edges_sha256"]:
        raise VisionDataError(f"{name}: edges changed during read")
    return data


def stratified_masks(labels, seed, fractions=(0.60, 0.20, 0.20)):
    """Fixed stratified split; exact per-class quotas, train/val >= 1.

    Per class of size n_c: n_train = floor(0.6 n_c), n_val = floor(0.2 n_c),
    n_test = remainder.  All quotas are positive for the audited datasets
    (minimum class size 10).
    """
    fr = np.asarray(fractions, dtype=float)
    if fr.shape != (3,) or not np.isclose(fr.sum(), 1.0):
        raise ValueError("fractions must sum to 1")
    labels = np.asarray(labels)
    n = len(labels)
    rng = np.random.default_rng(int(seed))
    train = np.zeros(n, dtype=bool)
    val = np.zeros(n, dtype=bool)
    test = np.zeros(n, dtype=bool)
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        idx = idx[rng.permutation(len(idx))]
        nc = len(idx)
        n_tr = max(1, int(np.floor(fr[0] * nc)))
        n_va = max(1, int(np.floor(fr[1] * nc)))
        if n_tr + n_va >= nc:
            n_va = max(0, nc - n_tr - 1)
        assign = np.array_split(idx, [n_tr, n_tr + n_va])
        train[assign[0]] = True
        val[assign[1]] = True
        test[assign[2]] = True
    if not (train.any() and val.any() and test.any()):
        raise ValueError("empty split")
    if (train & val).any() or (train & test).any() or (val & test).any():
        raise ValueError("split masks overlap")
    if not (train | val | test).all():
        raise ValueError("split masks do not cover all nodes")
    return {"train": train, "val": val, "test": test}


def split_fingerprint(masks):
    """Stable SHA-256 of the three index sets (order independent)."""
    payload = {k: np.flatnonzero(v).astype(int).tolist()
               for k, v in sorted(masks.items())}
    blob = json.dumps(payload, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def fit_standardizer(X, train_mask):
    """TRAIN-only mean/std; zero-variance dims get scale 1 and no noise."""
    X = np.asarray(X, dtype=np.float64)
    Xtr = X[train_mask]
    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0)
    zero_var = std < 1e-12
    scale = np.where(zero_var, 1.0, std)
    return {"mean": mean, "scale": scale, "zero_var": zero_var}


def standardize(X, standardizer):
    return (np.asarray(X, dtype=np.float64) - standardizer["mean"]) / \
        standardizer["scale"]


def noise_matrix(n, d, train_mask, zero_var, rng):
    """Unit standard-normal noise on non-train objects; zero-var cols untouched.

    In standardised coordinates this multiplies the declared sigma, so
    ``X_std + sigma * Z`` equals the protocol's
    ``(X + sigma * s_train * Z_raw - mean) / s_train``.
    """
    Z = rng.standard_normal((int(n), int(d)))
    Z[np.asarray(train_mask, dtype=bool)] = 0.0
    Z[:, np.asarray(zero_var, dtype=bool)] = 0.0
    return Z


def dataset_summary(data):
    deg = np.zeros(len(data.labels), dtype=np.int64)
    for e in data.edges:
        for u in e:
            deg[u] += 1
    sizes = np.array([len(e) for e in data.edges])
    return {
        "name": data.name,
        "n_objects": int(len(data.labels)),
        "n_edges": int(len(data.edges)),
        "n_classes": int(data.n_classes),
        "feature_dim": int(data.feature_dim),
        "n_extra_content_rows": int(data.n_extra_rows),
        "hyperedge_size_min": int(sizes.min()),
        "hyperedge_size_max": int(sizes.max()),
        "hyperedge_size_unique": sorted(int(v) for v in np.unique(sizes)),
        "object_degree_min": int(deg.min()),
        "object_degree_max": int(deg.max()),
        "class_count_min": int(np.bincount(data.labels).min()),
        "class_count_max": int(np.bincount(data.labels).max()),
        "content_sha256": data.content_sha256,
        "edges_sha256": data.edges_sha256,
    }
