import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "export_onnx.py"

assert SCRIPT_PATH.exists(), f"missing export script: {SCRIPT_PATH}"


def load_module():
    spec = importlib.util.spec_from_file_location("export_onnx", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExportOnnxTests(unittest.TestCase):
    def test_resolve_path_uses_repo_root_for_relative_paths(self):
        module = load_module()

        resolved = module.resolve_path("weight/best.pt")

        self.assertEqual(resolved, ROOT / "weight" / "best.pt")

    def test_default_onnx_output_path_lives_under_deploy_agx_zed_models(self):
        module = load_module()

        output = module.default_output_path(ROOT / "weight" / "best.pt")

        self.assertEqual(output, ROOT / "deploy" / "agx_zed" / "models" / "best.onnx")

    def test_build_export_kwargs_prefers_static_640_shape(self):
        module = load_module()

        kwargs = module.build_export_kwargs(imgsz=640, opset=17, dynamic=False, simplify=True)

        self.assertEqual(kwargs["format"], "onnx")
        self.assertEqual(kwargs["imgsz"], 640)
        self.assertEqual(kwargs["opset"], 17)
        self.assertFalse(kwargs["dynamic"])
        self.assertTrue(kwargs["simplify"])

    def test_export_model_invokes_ultralytics_export(self):
        fake_model = types.SimpleNamespace(export=lambda **kwargs: "ok")
        fake_yolo = lambda weights: fake_model
        fake_ultralytics = types.SimpleNamespace(YOLO=fake_yolo)

        with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
            module = load_module()
            with patch.object(module, "build_export_kwargs", return_value={"format": "onnx", "imgsz": 640, "path": "out.onnx"}) as build_kwargs:
                result = module.export_model(ROOT / "weight" / "best.pt", ROOT / "deploy" / "agx_zed" / "models" / "best.onnx", 640, 17, False, True)

        self.assertEqual(result, "ok")
        build_kwargs.assert_called_once_with(imgsz=640, opset=17, dynamic=False, simplify=True, output_path=ROOT / "deploy" / "agx_zed" / "models" / "best.onnx")


if __name__ == "__main__":
    unittest.main()
