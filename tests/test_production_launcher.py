import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "start_production.ps1"
CPU_SERVER = ROOT / "build" / "patched-cpu-noui" / "bin" / "Release" / "llama-server.exe"
CUDA_SERVER = ROOT / "build" / "patched-cuda-ninja3" / "bin" / "llama-server.exe"
REQUIRE_BINARIES = os.environ.get("CACHEFLOW_REQUIRE_PRODUCTION_BINARIES") == "1"


@unittest.skipUnless(
    CPU_SERVER.exists() or REQUIRE_BINARIES,
    "production launcher requires the patched CPU server",
)
class ProductionLauncherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.model_a = root / "model-a.gguf"
        self.model_b = root / "model-b.gguf"
        self.api_keys = root / "api-keys.txt"
        self.model_a.write_bytes(b"model-a")
        self.model_b.write_bytes(b"model-b")
        self.api_keys.write_text("production-secret\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def launch(self, model: Path | None = None, *extra: str) -> subprocess.CompletedProcess[str]:
        command = [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-ModelPath",
            str(model or self.model_a),
            "-ApiKeyFile",
            str(self.api_keys),
        ]
        if "-Backend" not in extra:
            command.extend(("-Backend", "cpu"))
        command.extend(("-PrintCommand", *extra))
        return subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )

    def identity(self, model: Path | None = None, *extra: str) -> str:
        result = self.launch(model, *extra)
        self.assertEqual(result.returncode, 0, result.stderr)
        return str(json.loads(result.stdout)["checkpoint_key"])

    def test_identity_is_stable_for_same_deployment(self) -> None:
        self.assertEqual(self.identity(), self.identity())

    def test_identity_cannot_cross_model_or_serving_shape(self) -> None:
        baseline = self.identity()
        variants = {
            self.identity(self.model_b),
            self.identity(None, "-ContextSize", "4096"),
            self.identity(None, "-Parallel", "2"),
            self.identity(None, "-CheckpointNamespace", "canary"),
        }
        self.assertEqual(len(variants), 4)
        self.assertNotIn(baseline, variants)

    @unittest.skipUnless(
        CUDA_SERVER.exists() or REQUIRE_BINARIES,
        "backend binding requires the patched CUDA server",
    )
    def test_identity_is_backend_bound(self) -> None:
        cpu = self.identity()
        result = self.launch(None, "-Backend", "cuda")
        self.assertEqual(result.returncode, 0, result.stderr)
        cuda = str(json.loads(result.stdout)["checkpoint_key"])
        self.assertNotEqual(cpu, cuda)

    def test_raw_checkpoint_key_override_is_rejected(self) -> None:
        result = self.launch(None, "-CheckpointKey", "unsafe-global-key")
        self.assertNotEqual(result.returncode, 0)

    def test_whitespace_only_api_key_is_rejected(self) -> None:
        self.api_keys.write_text("  \n\t\n", encoding="utf-8")
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("at least one non-empty key", result.stderr)


if __name__ == "__main__":
    unittest.main()
