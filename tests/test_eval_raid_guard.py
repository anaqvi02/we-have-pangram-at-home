"""Regression tests for the eval_raid.py class-balance guard.

The 2026-01-31 essay run silently evaluated an AI-only set and printed a
meaningless 0% FPR. RAID got the same trap: a bad domain/model filter (or
--max_samples cutting off before any human sample) would produce a single-class
"overall" result. These tests lock in the fix.

The overall evaluation requires both classes and exits 1 otherwise. The
per-model/per-attack breakdowns stay single-class on purpose (detection rate
= recall), so they must NOT raise.

Run standalone (no pytest needed):
    python tests/test_eval_raid_guard.py
"""
import glob
import importlib.util
import os
import sys
import types
from types import SimpleNamespace

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_PATH = os.path.join(REPO_ROOT, "scripts", "eval_raid.py")

# ---- stub heavy deps before importing the module ----
def _stub(name, attrs=None):
    m = types.ModuleType(name)
    m.__path__ = []
    for k, v in (attrs or {}).items():
        setattr(m, k, v)
    sys.modules.setdefault(name, m)
    return m

_torch = _stub("torch", {
    "no_grad": lambda: SimpleNamespace(__enter__=lambda s: None, __exit__=lambda s, *a: False)})
class _OOM(Exception):
    """Stand-in for torch.cuda.OutOfMemoryError in the stubbed torch."""

_cuda = _stub("torch.cuda", {
    "is_available": lambda: False,
    "empty_cache": lambda: None,
    "OutOfMemoryError": _OOM,
})
_backends = _stub("torch.backends")
_mps = _stub("torch.backends.mps", {"is_available": lambda: False, "is_built": lambda: False})
_torch.cuda, _torch.backends = _cuda, _backends
_backends.mps = _mps
_torch.nn = _stub("torch.nn", {"Module": type})
_stub("torch.utils")
_stub("torch.utils.data", {"DataLoader": object})
_stub("torch.nn")
_stub("numpy", {"array": lambda x, *a, **k: x, "unique": lambda x: x, "ndarray": type})
_stub("sklearn")
class _FakeCM:
    shape = (2, 2)
    def tolist(self):
        return [[5, 0], [0, 5]]
    def ravel(self):
        return [5, 0, 0, 5]

_stub("sklearn.metrics", {
    "accuracy_score": lambda a, b: 0.9, "precision_score": lambda a, b, **k: 0.9,
    "recall_score": lambda a, b, **k: 0.9, "f1_score": lambda a, b, **k: 0.9,
    "roc_auc_score": lambda a, b: 0.9, "confusion_matrix": lambda a, b: _FakeCM(),
    "classification_report": lambda *a, **k: ""})
_stub("tqdm", {"tqdm": lambda it, **k: it})
_stub("datasets", {"load_dataset": lambda *a, **k: FakeRaid()})
_stub("src.model", {"__path__": []})

_det = types.ModuleType("src.model.detector")
class _FakeDetector:
    def __init__(self):
        self.model = SimpleNamespace(eval=lambda: SimpleNamespace())
        self.tokenizer = None
        self.config = SimpleNamespace(DEVICE="cpu")
    @classmethod
    def load(cls, path):
        return cls()
_det.PangramDetector = _FakeDetector
sys.modules["src.model.detector"] = _det

_spec = importlib.util.spec_from_file_location("eval_raid_mod", EVAL_PATH)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)
# capture the real predict_batch: _install_stubs() replaces it with a lambda
_REAL_PREDICT_BATCH = mod.predict_batch

# dataset stub (streaming mode: filter() + iteration only)
class FakeRaid:
    def __init__(self, samples=None):
        self.samples = samples if samples is not None else []
    def filter(self, fn):
        return FakeRaid([s for s in self.samples if fn(s)])
    def __iter__(self):
        return iter(self.samples)


def _sample(is_human):
    return {
        "generation": "x" * 200,
        "model": None if is_human else "chatgpt",
        "domain": "news",
        # RAID semantics: humans carry attack=None, clean AI carries
        # attack="none", attacked AI carries the attack name.
        "attack": None if is_human else "none",
        "decoding": "sample",
    }


def _install_stubs():
    mod.load_dataset = lambda *a, **k: FakeRaid()
    mod.predict_batch = lambda detector, texts, batch_size=32: (
        [1] * len(texts), [0.9] * len(texts))


def _cleanup():
    for f in glob.glob("/tmp/raid_eval_*.json"):
        os.remove(f)


def test_syntax():
    compile(open(EVAL_PATH, encoding="utf-8").read(), EVAL_PATH, "exec")


def test_ai_only_overall_exits():
    _install_stubs()
    mod.load_dataset = lambda *a, **k: FakeRaid([_sample(False) for _ in range(20)])
    try:
        mod.run_full_evaluation(model_path="dummy", max_samples=100, output_dir="/tmp")
    except SystemExit as e:
        assert e.code == 1, f"expected exit 1, got {e.code}"
        return
    assert False, "expected SystemExit(1), ran to completion"


def test_human_only_overall_exits():
    _install_stubs()
    mod.load_dataset = lambda *a, **k: FakeRaid([_sample(True) for _ in range(20)])
    try:
        mod.run_full_evaluation(model_path="dummy", max_samples=100, output_dir="/tmp")
    except SystemExit as e:
        assert e.code == 1, f"expected exit 1, got {e.code}"
        return
    assert False, "expected SystemExit(1), ran to completion"


def test_balanced_overall_proceeds():
    _install_stubs()
    samples = [_sample(i % 2 == 0) for i in range(20)]  # 10 human, 10 AI
    mod.load_dataset = lambda *a, **k: FakeRaid(samples)
    results = mod.run_full_evaluation(model_path="dummy", max_samples=100, output_dir="/tmp")
    overall = results["overall"]
    assert overall["human_samples"] == 10, f"expected 10 human, got {overall['human_samples']}"
    assert overall["ai_samples"] == 10, f"expected 10 AI, got {overall['ai_samples']}"
    _cleanup()


def test_breakdown_single_class_stays_legal():
    # per-model/per-attack subsets are single-class by design; must not raise
    _install_stubs()
    ai_only = FakeRaid([_sample(False) for _ in range(20)])
    metrics = mod.evaluate_subset(
        _FakeDetector(), ai_only, max_samples=100, require_both_classes=False)
    assert metrics["ai_samples"] == 20, f"expected 20 AI, got {metrics['ai_samples']}"
    assert metrics["human_samples"] == 0


def test_no_attacks_filters_attacked_samples():
    # clean-only run must keep humans + clean AI, drop attacked AI
    _install_stubs()
    samples = [_sample(True) for _ in range(10)]  # humans, attack=None
    samples += [_sample(False) for _ in range(10)]  # clean AI, attack="none"
    for _ in range(10):  # attacked AI
        s = _sample(False)
        s["attack"] = "homoglyph"
        samples.append(s)
    mod.load_dataset = lambda *a, **k: FakeRaid(samples)
    results = mod.run_full_evaluation(
        model_path="dummy", max_samples=100, output_dir="/tmp", no_attacks=True)
    overall = results["overall"]
    assert overall["ai_samples"] == 10, f"expected 10 clean AI, got {overall['ai_samples']}"
    assert overall["human_samples"] == 10, f"expected 10 human, got {overall['human_samples']}"
    assert results["config"]["no_attacks"] is True
    _cleanup()


def test_predict_batch_halves_batch_on_oom():
    # first chunk OOMs -> batch halves, same texts retried, all processed
    calls = []

    def fake_chunk(model, tokenizer, chunk, device):
        calls.append(len(chunk))
        if len(calls) == 1:
            raise _OOM()
        return [1] * len(chunk), [0.9] * len(chunk)

    mod._predict_chunk = fake_chunk
    mod.predict_batch = _REAL_PREDICT_BATCH  # undo the _install_stubs() lambda
    preds, probs = mod.predict_batch(_FakeDetector(), ["x"] * 100, batch_size=32)
    assert calls[0] == 32, f"first chunk {calls[0]} != 32"
    assert max(calls[1:]) <= 16, f"post-OOM chunks not halved: {calls}"
    assert sum(calls[1:]) == 100, f"processed {sum(calls[1:])} != 100: {calls}"
    assert len(preds) == len(probs) == 100


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failures else 0)
