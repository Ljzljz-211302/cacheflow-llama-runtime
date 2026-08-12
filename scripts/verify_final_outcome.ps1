$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $projectRoot "src"
Push-Location $projectRoot
try {
    python scripts\build_final_outcome.py --check
    if ($LASTEXITCODE -ne 0) { throw "final outcome validation failed" }
    python scripts\validate_final_evidence.py
    if ($LASTEXITCODE -ne 0) { throw "formal evidence closure failed" }
    python -m unittest tests.test_final_outcome -v
    if ($LASTEXITCODE -ne 0) { throw "final outcome tests failed" }
}
finally {
    Pop-Location
}
