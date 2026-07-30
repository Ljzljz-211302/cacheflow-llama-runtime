param([switch]$Full)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $projectRoot "src"

Push-Location $projectRoot
try {
    python -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) {
        throw "unit tests failed"
    }

    python -m compileall -q src scripts tests
    if ($LASTEXITCODE -ne 0) {
        throw "compileall failed"
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

    & "runtime\bin\llama-bench.exe" --list-devices
    if ($LASTEXITCODE -ne 0) {
        throw "llama-bench device check failed"
    }

    if ($Full) {
        powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build_patched_server.ps1
        if ($LASTEXITCODE -ne 0) { throw "patched server build failed" }
        powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build_cuda_kv.ps1
        if ($LASTEXITCODE -ne 0) { throw "CUDA KV backend build failed" }
        python scripts\run_kv_block_smoke.py --mode share
        if ($LASTEXITCODE -ne 0) { throw "resident prefix sharing smoke failed" }
        python scripts\run_kv_block_smoke.py --mode preempt --port 8108
        if ($LASTEXITCODE -ne 0) { throw "preempt/restore smoke failed" }
        python scripts\run_engine_ab.py
        if ($LASTEXITCODE -ne 0) { throw "engine scheduler A/B failed" }
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
