import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "infer_volleyball.py"

assert SCRIPT_PATH.exists(), f"missing infer script: {SCRIPT_PATH}"


def load_module():
    fake_ultralytics = types.SimpleNamespace(YOLO=object)
    with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
        spec = importlib.util.spec_from_file_location("infer_volleyball", SCRIPT_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class InferVolleyballTests(unittest.TestCase):
    def test_normalize_source_preserves_webcam_index(self):
        module = load_module()

        source = module.normalize_source("0")

        self.assertEqual(source, 0)

    def test_normalize_source_preserves_stream_url(self):
        module = load_module()

        source = module.normalize_source("rtsp://192.168.31.10/live")

        self.assertEqual(source, "rtsp://192.168.31.10/live")

    def test_normalize_source_resolves_relative_local_path(self):
        module = load_module()

        source = module.normalize_source("data/sample.jpg")

        self.assertEqual(source, str((ROOT / "data" / "sample.jpg").resolve()))


if __name__ == "__main__":
    unittest.main()
