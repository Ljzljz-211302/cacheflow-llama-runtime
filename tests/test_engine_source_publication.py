import configparser
import json
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class EngineSourcePublicationTest(unittest.TestCase):
    def test_submodule_and_manifest_pin_same_engine_snapshot(self) -> None:
        modules = configparser.ConfigParser()
        modules.read(ROOT / ".gitmodules", encoding="utf-8")
        section = modules["submodule \"vendor/llama.cpp\""]
        self.assertEqual(section["path"], "vendor/llama.cpp")
        self.assertEqual(
            section["url"],
            "https://github.com/Ljzljz-211302/cacheflow-llama-runtime.git",
        )
        self.assertEqual(section["branch"], "engine/llama.cpp")

        manifest = json.loads((ROOT / "config/artifacts.json").read_text(encoding="utf-8"))
        entry = subprocess.run(
            ["git", "ls-files", "-s", "vendor/llama.cpp"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().split()
        self.assertEqual(entry[0], "160000")
        self.assertEqual(entry[1], manifest["llama_cpp"]["commit"])
        self.assertEqual(manifest["llama_cpp"]["branch"], "engine/llama.cpp")

    def test_audit_patches_remain_available(self) -> None:
        for name in (
            "0001-cache-aware-slot-scheduler.patch",
            "0002-public-workload-routing.patch",
        ):
            self.assertTrue((ROOT / "patches" / name).is_file())


if __name__ == "__main__":
    unittest.main()
