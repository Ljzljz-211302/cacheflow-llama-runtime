param([switch]$Full)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = ((Join-Path $projectRoot "src"), (Join-Path $projectRoot "prototypes")) -join ";"

Push-Location $projectRoot
try {
    python -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) {
        throw "unit tests failed"
    }

    python -m compileall -q src prototypes scripts tests
    if ($LASTEXITCODE -ne 0) {
        throw "compileall failed"
    }

    python scripts\audit_architecture.py
    if ($LASTEXITCODE -ne 0) {
        throw "architecture ownership audit failed"
    }

    $manifest = Get-Content -Raw -Encoding utf8 "config\artifacts.json" | ConvertFrom-Json
    foreach ($artifact in $manifest.artifacts) {
        $path = Join-Path $projectRoot $artifact.path
        if (-not (Test-Path -LiteralPath $path)) {
            throw "missing artifact: $($artifact.path)"
        }
        $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash
        if ($hash -ne $artifact.sha256) {
            throw "checksum mismatch: $($artifact.path)"
        }
    }
    Write-Host "Artifact checksums passed."

    if ($Full) {
        & git -C "vendor\llama.cpp" apply --reverse --check "..\..\patches\0001-cache-aware-slot-scheduler.patch"
        if ($LASTEXITCODE -ne 0) {
            throw "engine patch is not the exact reversible diff for the pinned fork"
        }
        Write-Host "Pinned upstream patch reversibility passed."
    }

    & "runtime\bin\llama-bench.exe" --list-devices
    if ($LASTEXITCODE -ne 0) {
        throw "llama-bench device check failed"
    }

    if ($Full) {
        powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build_patched_server.ps1
        if ($LASTEXITCODE -ne 0) { throw "patched server build failed" }
        powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build_upstream_baseline.ps1
        if ($LASTEXITCODE -ne 0) { throw "same-toolchain upstream baseline build failed" }
        powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build_cuda_kv.ps1 -Sanitize
        if ($LASTEXITCODE -ne 0) { throw "CUDA KV backend build or Compute Sanitizer failed" }
        python scripts\run_kv_block_smoke.py --mode share
        if ($LASTEXITCODE -ne 0) { throw "resident prefix sharing smoke failed" }
        python scripts\run_kv_block_smoke.py --mode preempt --port 8108
        if ($LASTEXITCODE -ne 0) { throw "preempt/restore smoke failed" }
        python scripts\run_cuda_tensor_adapter_smoke.py
        if ($LASTEXITCODE -ne 0) { throw "real CUDA tensor adapter smoke failed" }
        python scripts\run_cuda_swap_server_smoke.py
        if ($LASTEXITCODE -ne 0) { throw "real CUDA server swap smoke failed" }
        python scripts\run_kv_store_server_smoke.py
        if ($LASTEXITCODE -ne 0) { throw "production host/file KV store and failure fallback smoke failed" }
        python scripts\run_runtime_fault_injection.py
        if ($LASTEXITCODE -ne 0) { throw "runtime fault injection failed" }
        python scripts\run_openai_compat_smoke.py
        if ($LASTEXITCODE -ne 0) { throw "OpenAI stream/non-stream compatibility failed" }
        python scripts\run_serving_control_smoke.py
        if ($LASTEXITCODE -ne 0) { throw "cancel/deadline/backpressure smoke failed" }
        python scripts\run_upstream_compat.py
        if ($LASTEXITCODE -ne 0) { throw "same-toolchain upstream compatibility failed" }
        python scripts\run_model_acceptance_matrix.py
        if ($LASTEXITCODE -ne 0) { throw "real model/backend/concurrency/context matrix failed" }
        python scripts\run_cuda_kv_benchmark.py
        if ($LASTEXITCODE -ne 0) { throw "CUDA KV transport benchmark failed" }
        python scripts\run_cuda_cow_benchmark.py
        if ($LASTEXITCODE -ne 0) { throw "CUDA COW P95 benchmark failed" }
        python scripts\run_adaptive_prefill_ab.py
        if ($LASTEXITCODE -ne 0) { throw "adaptive prefill CPU/CUDA A/B failed" }
        python scripts\run_adaptive_speculation_ab.py
        if ($LASTEXITCODE -ne 0) { throw "adaptive speculation CPU/CUDA A/B failed" }
        python scripts\run_engine_ab.py
        if ($LASTEXITCODE -ne 0) { throw "engine scheduler A/B failed" }
        python scripts\run_mixed_workload.py --backend both --trials 3
        if ($LASTEXITCODE -ne 0) { throw "mixed prefill/decode CPU/CUDA workload failed" }
        python scripts\run_benefit_gating_ab.py --backend both --trials 10
        if ($LASTEXITCODE -ne 0) { throw "conservative benefit gating/oracle acceptance failed" }
        powershell -NoProfile -ExecutionPolicy Bypass -File scripts\profile_engine.ps1
        if ($LASTEXITCODE -ne 0) { throw "production engine trace/flame chart failed" }
        python scripts\run_prefill_ab.py
        if ($LASTEXITCODE -ne 0) { throw "chunked prefill A/B failed" }
        python scripts\run_scheduler_trace.py
        if ($LASTEXITCODE -ne 0) { throw "scheduler trace simulation failed" }
        python scripts\run_benchmarks.py
        if ($LASTEXITCODE -ne 0) { throw "offline benchmark failed" }
        python scripts\run_server_benchmark.py
        if ($LASTEXITCODE -ne 0) { throw "server benchmark failed" }
        python scripts\run_quality.py
        if ($LASTEXITCODE -ne 0) { throw "quality benchmark failed" }
        python scripts\generate_report.py
        if ($LASTEXITCODE -ne 0) { throw "report generation failed" }
    }
}
finally {
    Pop-Location
}
