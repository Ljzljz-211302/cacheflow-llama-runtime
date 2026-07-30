$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$manifestPath = Join-Path $projectRoot "config\artifacts.json"
$manifest = Get-Content -Raw -Encoding utf8 $manifestPath | ConvertFrom-Json

function Test-ArtifactHash {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$ExpectedHash
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        return $false
    }
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash
    return $actual -eq $ExpectedHash
}

function Get-Artifact {
    param([Parameter(Mandatory = $true)]$Artifact)

    $destination = Join-Path $projectRoot $Artifact.path
    $parent = Split-Path -Parent $destination
    New-Item -ItemType Directory -Force -Path $parent | Out-Null

    if (Test-ArtifactHash -Path $destination -ExpectedHash $Artifact.sha256) {
        Write-Host "[ok] $($Artifact.name)"
        return
    }

    if (Test-Path -LiteralPath $destination) {
        $suffix = Get-Date -Format "yyyyMMddHHmmss"
        Move-Item -LiteralPath $destination -Destination "$destination.invalid-$suffix"
    }
    $partial = "$destination.part"
    Write-Host "[download] $($Artifact.name)"
    & curl.exe --fail --location --retry 5 --retry-all-errors -C - --output $partial $Artifact.url
    if ($LASTEXITCODE -ne 0) {
        throw "download failed: $($Artifact.name)"
    }
    if (-not (Test-ArtifactHash -Path $partial -ExpectedHash $Artifact.sha256)) {
        throw "checksum mismatch: $($Artifact.name)"
    }
    Move-Item -Force -LiteralPath $partial -Destination $destination
}

$vendorRoot = Join-Path $projectRoot "vendor"
$sourceRoot = Join-Path $vendorRoot "llama.cpp"
New-Item -ItemType Directory -Force -Path $vendorRoot | Out-Null
if (-not (Test-Path -LiteralPath (Join-Path $sourceRoot ".git"))) {
    & git clone --depth 1 --branch $manifest.llama_cpp.tag $manifest.llama_cpp.repository $sourceRoot
    if ($LASTEXITCODE -ne 0) {
        throw "failed to clone llama.cpp"
    }
}
$actualCommit = (& git -C $sourceRoot rev-parse HEAD).Trim()
if ($actualCommit -ne $manifest.llama_cpp.commit) {
    throw "llama.cpp commit mismatch: expected $($manifest.llama_cpp.commit), got $actualCommit"
}
Write-Host "[ok] llama.cpp $($manifest.llama_cpp.tag) ($actualCommit)"

foreach ($artifact in $manifest.artifacts) {
    Get-Artifact -Artifact $artifact
}

$archives = $manifest.artifacts | Where-Object { $_.extract_to }
foreach ($archive in $archives) {
    $archivePath = Join-Path $projectRoot $archive.path
    $extractPath = Join-Path $projectRoot $archive.extract_to
    New-Item -ItemType Directory -Force -Path $extractPath | Out-Null
    Expand-Archive -Force -LiteralPath $archivePath -DestinationPath $extractPath
}

$bench = Join-Path $projectRoot "runtime\bin\llama-bench.exe"
if (-not (Test-Path -LiteralPath $bench)) {
    throw "llama-bench.exe missing after extraction"
}
& $bench --list-devices
if ($LASTEXITCODE -ne 0) {
    throw "llama.cpp runtime validation failed"
}

Write-Host "Bootstrap complete."
