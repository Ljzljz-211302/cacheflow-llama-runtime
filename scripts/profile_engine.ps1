param([switch]$SystemSample)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$raw = Join-Path $projectRoot "results\raw"
New-Item -ItemType Directory -Force -Path $raw | Out-Null

Push-Location $projectRoot
try {
    if ($SystemSample) {
        $status = (& wpr.exe -status | Out-String)
        if ($status -notmatch "WPR is not recording") {
            throw "WPR is already recording; refusing to disturb the active profiling session"
        }
        & wpr.exe -start CPU -filemode
        if ($LASTEXITCODE -ne 0) {
            throw "WPR system sampling requires SeSystemProfilePrivilege; use the default in-process trace"
        }
    }

    try {
        python scripts\run_mixed_workload.py --backend cpu --policy cacheflow --trials 1 `
            --output-prefix profile_mixed --engine-trace
        if ($LASTEXITCODE -ne 0) { throw "profile workload failed" }
    }
    finally {
        if ($SystemSample) {
            & wpr.exe -stop (Join-Path $raw "cacheflow-engine-cpu.etl")
        }
    }

    python scripts\render_engine_flame.py
    if ($LASTEXITCODE -ne 0) { throw "engine flame chart rendering failed" }
}
finally {
    Pop-Location
}

Write-Host "Production engine trace and flame chart generated."
