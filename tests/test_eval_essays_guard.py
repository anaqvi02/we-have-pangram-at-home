"""Regression tests for the eval_essays.py single-class guard.

The 2026-01-31 benchmark run silently evaluated an AI-only set and printed
"FPR 0.0%" that meant nothing: the human sources failed to load and the
script kept going. These tests lock in the fix — run_benchmark() must exit 1
when either class is empty, and must never write a results file in that case.

Run standalone (no pytest needed):
    python tests/test_eval_essays_guard.py

Heavy deps (torch, datasets, sklearn, model classes) are stubbed; only the
control flow of run_benchmark() is exercised.
"""
import glob
import importlib.util
import os
import sys
import types
from types import SimpleNamespace

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_PATH = os.path.join(REPO_ROOT, "scripts", "eval_essays.py")

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
_cuda = _stub("torch.cuda", {"is_available": lambda: False})
_backends = _stub("torch.backends")
_mps = _stub("torch.backends.mps", {"is_available": lambda: False, "is_built": lambda: False})
_torch.cuda, _torch.backends = _cuda, _backends
_backends.mps = _mps
_stub("torch.utils")
_stub("torch.utils.data", {"DataLoader": object})
_stub("torch.nn")
_stub("numpy", {"array": lambda x, *a, **k: x, "unique": lambda x: x, "ndarray": type})
_stub("sklearn")
_stub("sklearn.metrics", {
    "roc_curve": lambda *a: ([], [], []),
    "accuracy_score": lambda a, b: 0.0, "precision_score": lambda a, b, **k: 0.0,
    "recall_score": lambda a, b, **k: 0.0, "f1_score": lambda a, b, **k: 0.0,
    "roc_auc_score": lambda a, b: 0.0, "confusion_matrix": lambda a, b: [[0, 0], [0, 0]]})
_stub("tqdm", {"tqdm": lambda it, **k: it})
_stub("datasets", {"load_dataset": lambda *a, **k: None})
_stub("src.model", {"__path__": []})

_det = types.ModuleType("src.model.detector")
class _FakeDetector:
    def __init__(self):
        self.model = SimpleNamespace(eval=lambda: SimpleNamespace())
        self.tokenizer = None
        self.config = None
    @classmethod
    def load(cls, path):
        return cls()
_det.PangramDetector = _FakeDetector
sys.modules["src.model.detector"] = _det

_spec = importlib.util.spec_from_file_location("eval_essays_mod", EVAL_PATH)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

_FAKE_METRICS = {
    "num_samples": 20, "accuracy": 0.9, "precision": 0.9, "recall": 0.9,
    "f1": 0.9, "roc_auc": 0.9, "confusion_matrix": [[5, 0], [0, 5]],
}
HUMAN_NAMES = {"HC3-Human", "Reddit-Writing"}


def _run_case(texts_fn, expect_exit):
    mod.load_benchmark_source = lambda source, max_samples: texts_fn(source)
    mod.evaluate_texts = lambda detector, texts, labels, **k: dict(_FAKE_METRICS)
    try:
        mod.run_benchmark(model_path="dummy", samples_per_source=5, output_dir="/tmp")
    except SystemExit as e:
        assert expect_exit, f"unexpected SystemExit({e.code})"
        assert e.code == 1, f"expected exit code 1, got {e.code}"
        return
    assert not expect_exit, "expected SystemExit(1), script ran to completion"
    # clean the results file written by the balanced case
    for f in glob.glob("/tmp/essay_eval_*.json"):
        os.remove(f)


def test_syntax():
    compile(open(EVAL_PATH, encoding="utf-8").read(), EVAL_PATH, "exec")


def test_all_sources_fail_exits():
    _run_case(lambda s: [], expect_exit=True)


def test_balanced_classes_proceeds():
    _run_case(lambda s: ["x" * 200] * 5, expect_exit=False)


def test_human_sources_fail_exits():
    # the 2026-01-31 scenario: AI loads, human sources fail
    _run_case(lambda s: [] if s["name"] in HUMAN_NAMES else ["x" * 200] * 5,
              expect_exit=True)


def test_all_sources_have_data_files():
    # HF removed dataset loading scripts (datasets 3.x); every benchmark
    # source must load from the datasets-server auto-converted parquet.
    for group in mod.BENCHMARK_SOURCES.values():
        for source in group:
            assert source.get("data_files"), f"{source['name']} missing data_files"


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
