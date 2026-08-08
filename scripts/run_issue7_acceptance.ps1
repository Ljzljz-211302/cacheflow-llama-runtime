param([switch]$IncludeSanitizer)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot

Push-Location $projectRoot
try {
    python scripts\run_production_paged_experiment.py --validate-only
    if ($LASTEXITCODE -ne 0) { throw "formal production Paged artifact validation failed" }

    python scripts\run_production_paged_journey.py
    if ($LASTEXITCODE -ne 0) { throw "Direct/Paged production journey failed" }

    python scripts\run_cuda_tensor_adapter_smoke.py
    if ($LASTEXITCODE -ne 0) { throw "real Remap CUDA journey failed" }

    python scripts\run_kv_pressure_fallback_journey.py
    if ($LASTEXITCODE -ne 0) { throw "KV pressure fallback journey failed" }

    python scripts\run_paged_cuda_failure_journey.py
    if ($LASTEXITCODE -ne 0) { throw "Paged late-CUDA-failure recovery journey failed" }

    if ($IncludeSanitizer) {
        python scripts\run_production_paged_journey.py --memcheck
        if ($LASTEXITCODE -ne 0) { throw "production Paged Compute Sanitizer journey failed" }
    }

    Write-Host "Issue #7 Direct, Remap, Paged, pressure fallback, and failure-recovery journeys passed."
}
finally {
    Pop-Location
}
