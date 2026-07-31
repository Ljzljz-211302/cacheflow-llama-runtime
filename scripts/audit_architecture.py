from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "vendor/llama.cpp/tools/server"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    context = (SERVER / "server-context.cpp").read_text(encoding="utf-8")
    engine_h = (SERVER / "server-inference-engine.h").read_text(encoding="utf-8")
    memory_h = (ROOT / "vendor/llama.cpp/src/llama-memory.h").read_text(encoding="utf-8")

    require("server_inference_engine inference_engine" in context,
            "server_context must compose the inference engine")
    require("inference_engine.step(" in context,
            "production update loop must enter through engine.step")
    require("plan_execute_iteration(server_inference_iteration & iteration)" in context,
            "context callback must receive the engine-owned transaction")
    require("bool step(Prepare && prepare, PlanExecute && plan_execute)" in engine_h,
            "engine must own prepare/plan-execute/commit ordering")
    require("std::unique_ptr<server_kv_runtime> kv_runtime_" in engine_h,
            "engine must own KV runtime")
    require("std::unique_ptr<server_kv_swap_store> swap_store_" in engine_h,
            "engine must own swap store")
    require("server_speculation_controller speculation_" in engine_h,
            "engine must own speculation controller")
    require("virtual bool cacheflow_set_block_size(uint32_t) { return false; }" in memory_h,
            "unsupported memory types must explicitly reject physical block capability")
    require("physical_block_capability" in context and "logical-only" in context,
            "production server must expose physical-KV capability downgrade")
    retired_source = ROOT / "src/cacheflow"
    require(not retired_source.exists() or not any(retired_source.rglob("*.py")),
            "retired Python control-plane source must not remain in production src")
    require((ROOT / "prototypes/cacheflow").is_dir(),
            "retired Python control plane must remain auditable under prototypes")

    print("architecture ownership, transaction order, capability fallback, and prototype boundary passed")


if __name__ == "__main__":
    main()
