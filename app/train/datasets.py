"""Generic dataset layer — load ANY HuggingFace classification dataset.

No whitelist. The research agent proposes a dataset (id + optional config); this
module loads it and AUTO-DETECTS its structure:
  - the text field(s)  (text / sentence / content / ... or the first string column;
    multiple fields, e.g. NLI premise+hypothesis, are joined)
  - the label field    (a ClassLabel feature, or label / class / category / ...)
  - the label names    (from ClassLabel.names, or the unique values found)

The agent may pass explicit field hints; we use them when valid, else detect.
Falls back gracefully and raises a clear error the loop can surface to the UI.

Returns:
  label_names: [str, ...]
  pools:       {label_name: [text, ...]}      (grouped, so weak labels oversample)
  eval:        [{"text":..., "label":label_name}, ...]
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DatasetSpec:
    """What the agent decided to train on. Field hints are optional."""
    id: str
    config: str | None = None
    text_fields: list[str] = field(default_factory=list)   # [] -> auto-detect
    label_field: str | None = None                          # None -> auto-detect

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetSpec":
        tf = d.get("text_fields") or ([d["text_field"]] if d.get("text_field") else [])
        return cls(
            id=_canonical_id(d["id"]), config=d.get("config"),
            text_fields=list(tf), label_field=d.get("label_field"),
        )


# Legacy bare names → their canonical namespaced ids (the Hub now requires owner/name).
_ID_ALIASES = {
    "lex_glue": "coastalcph/lex_glue",
    "ag_news": "fancyzhx/ag_news",
    "emotion": "dair-ai/emotion",
    "banking77": "PolyAI/banking77",
    "tweet_eval": "cardiffnlp/tweet_eval",
    "sst2": "stanfordnlp/sst2",
    "scicite": "allenai/scicite",
    "ade_corpus_v2": "ade-benchmark-corpus/ade_corpus_v2",
}


def _canonical_id(ds_id: str) -> str:
    """Map a bare/legacy dataset id to a namespaced one when we know it."""
    ds_id = (ds_id or "").strip()
    return _ID_ALIASES.get(ds_id, ds_id)



# Common names, tried in order, when auto-detecting.
_TEXT_HINTS = ["text", "sentence", "sentence1", "content", "tweet", "review",
               "comment", "document", "question", "sms", "premise", "body", "title"]
_LABEL_HINTS = ["label", "labels", "class", "category", "target", "intent",
                "emotion", "sentiment", "topic", "stars", "rating"]
_EVAL_SPLITS = ["validation", "valid", "dev", "test"]


def _detect_text_fields(features, provided: list[str]) -> list[str]:
    from datasets import Value
    string_cols = [n for n, f in features.items()
                   if isinstance(f, Value) and f.dtype == "string"]
    if provided:
        good = [c for c in provided if c in features]
        if good:
            return good
    hits = [h for h in _TEXT_HINTS if h in string_cols]
    if hits:
        # keep paired fields together (sentence1/sentence2, premise/hypothesis)
        pairs = {"sentence1": "sentence2", "premise": "hypothesis", "question": "sentence"}
        first = hits[0]
        if first in pairs and pairs[first] in features:
            return [first, pairs[first]]
        return [first]
    return string_cols[:1]


def _detect_label_field(features, provided: str | None) -> str:
    from datasets import ClassLabel
    if provided and provided in features:
        return provided
    for n, f in features.items():
        if isinstance(f, ClassLabel):
            return n
    for h in _LABEL_HINTS:
        if h in features:
            return h
    raise ValueError(
        f"could not find a label column (columns: {list(features)}). "
        "Have the agent pass label_field."
    )


def _label_namer(feature, observed: list):
    """Return a function int/str -> name, plus the ordered name list."""
    from datasets import ClassLabel
    if isinstance(feature, ClassLabel):
        names = list(feature.names)
        return (lambda v: names[v] if isinstance(v, int) and 0 <= v < len(names) else None), names
    # Non-ClassLabel: derive names from observed values (stable order).
    uniq = []
    for v in observed:
        s = str(v)
        if s not in uniq:
            uniq.append(s)
    uniq.sort()
    return (lambda v: str(v)), uniq


def _join_text(row, text_fields: list[str]) -> str:
    parts = [str(row.get(c, "")).strip() for c in text_fields]
    return " ".join(p for p in parts if p).strip()


def peek_sample(spec: DatasetSpec, *, n_per_label: int = 3, max_examples: int = 18) -> dict:
    """Load a SMALL slice and return a sample across categories — for validation.

    Returns {labels, counts, samples:[{label,text}], text_fields, label_field}.
    Cheap (loads ~400 rows). Raises if the dataset can't be read/understood; the
    caller treats that as "couldn't validate" and proceeds.
    """
    from datasets import load_dataset, get_dataset_split_names
    try:
        splits = get_dataset_split_names(spec.id, spec.config)
    except Exception:
        splits = ["train"]
    split = "train" if "train" in splits else splits[0]
    try:
        raw = load_dataset(spec.id, spec.config, split=f"{split}[:400]")
    except Exception:
        ds = load_dataset(spec.id, spec.config, split=split)
        raw = ds.select(range(min(400, len(ds))))

    feats = raw.features
    text_fields = _detect_text_fields(feats, spec.text_fields)
    label_field = _detect_label_field(feats, spec.label_field)   # raises if none
    observed = [raw[i][label_field] for i in range(min(len(raw), 400))]
    name_of, label_names = _label_namer(feats[label_field], observed)

    counts: dict[str, int] = {}
    samples: list[dict] = []
    per: dict[str, int] = {}
    for row in raw:
        nm = name_of(row[label_field])
        if not nm:
            continue
        counts[nm] = counts.get(nm, 0) + 1
        if per.get(nm, 0) < n_per_label and len(samples) < max_examples:
            txt = _join_text(row, text_fields)
            if txt:
                samples.append({"label": nm, "text": txt[:200]})
                per[nm] = per.get(nm, 0) + 1
    return {"labels": [n for n in label_names if n], "counts": counts,
            "samples": samples, "text_fields": text_fields, "label_field": label_field}


def load_pools(
    spec: DatasetSpec,
    *,
    max_per_label_train: int = 200,
    max_eval: int = 240,
):
    """Load + auto-detect a dataset. See module docstring for return shape."""
    from datasets import load_dataset, get_dataset_split_names

    # --- choose splits -------------------------------------------------
    try:
        splits = get_dataset_split_names(spec.id, spec.config)
    except Exception:
        splits = ["train"]
    train_split = "train" if "train" in splits else splits[0]
    val_split = next((s for s in ["validation", "valid", "dev"] if s in splits and s != train_split), None)
    test_split = "test" if ("test" in splits and "test" != train_split) else None

    # Load a bounded slice of train so huge datasets don't blow up.
    def _load(split, n):
        try:
            return load_dataset(spec.id, spec.config, split=f"{split}[:{n}]")
        except Exception:
            ds = load_dataset(spec.id, spec.config, split=split)
            return ds.select(range(min(n, len(ds))))

    train_raw = _load(train_split, max_per_label_train * 12)
    features = train_raw.features

    text_fields = _detect_text_fields(features, spec.text_fields)
    if not text_fields:
        raise ValueError(f"no text column found (columns: {list(features)})")
    label_field = _detect_label_field(features, spec.label_field)

    observed = [train_raw[i][label_field] for i in range(min(len(train_raw), 2000))]
    name_of, label_names = _label_namer(features[label_field], observed)
    # keep deterministic, drop any that never resolve
    label_names = [n for n in label_names if n is not None]

    # --- group train by label -----------------------------------------
    pools: dict[str, list[str]] = {n: [] for n in label_names}
    for row in train_raw:
        name = name_of(row[label_field])
        if name in pools and len(pools[name]) < max_per_label_train:
            txt = _join_text(row, text_fields)
            if txt:
                pools[name].append(txt)
    pools = {k: v for k, v in pools.items() if v}        # drop empty labels
    label_names = list(pools.keys())

    # --- validation + held-out TEST sets -------------------------------
    # Validation = what the agent steers on each round.
    # Test       = scored once at the end; the agent NEVER sees it.
    # Both are kept strictly disjoint from train (and from each other).
    per_label_eval = max(8, max_eval // max(len(label_names), 1))

    def grab(split):
        by = {n: [] for n in label_names}
        raw = _load(split, per_label_eval * len(label_names) * 8)
        for row in raw:
            name = name_of(row[label_field])
            if name in by and len(by[name]) < per_label_eval:
                txt = _join_text(row, text_fields)
                if txt:
                    by[name].append(txt)
        return by

    def carve(per_label):
        """Pull per_label examples from each train pool tail and REMOVE them."""
        out = {}
        for name in label_names:
            pool = pools[name]
            take = min(per_label, max(0, len(pool) - 5))   # leave >=5 for training
            out[name] = pool[-take:] if take else []
            if take:
                pools[name] = pool[:-take]
        return out

    if val_split and test_split:                 # real splits for both (e.g. emotion)
        val_by, test_by = grab(val_split), grab(test_split)
    elif test_split:                             # one real held-out → use as TEST
        test_by, val_by = grab(test_split), carve(per_label_eval)
    elif val_split:                              # one real held-out → use as VAL
        val_by, test_by = grab(val_split), carve(per_label_eval)
    else:                                        # none → carve both from train (disjoint)
        val_by, test_by = carve(per_label_eval), carve(per_label_eval)

    val_examples = [{"text": t, "label": n} for n, ts in val_by.items() for t in ts]
    test_examples = [{"text": t, "label": n} for n, ts in test_by.items() for t in ts]
    return label_names, pools, val_examples, test_examples
