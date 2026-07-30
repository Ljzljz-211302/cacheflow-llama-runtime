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
