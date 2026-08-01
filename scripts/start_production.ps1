param(
    [Parameter(Mandatory = $true)][string]$ModelPath,
    [Parameter(Mandatory = $true)][string]$ApiKeyFile,
    [string]$StateDirectory = "runtime\state",
    [ValidateSet("cpu", "cuda")][string]$Backend = "cuda",
    [ValidatePattern("^[A-Za-z0-9._-]+$")][string]$InstanceId = "default",
    [ValidateRange(1, 65535)][int]$Port = 8080,
    [ValidateRange(512, 1048576)][int]$ContextSize = 8192,
    [ValidateRange(1, 128)][int]$Parallel = 4,
    [ValidateRange(1, 256)][int]$Threads = 8,
    [string]$CheckpointKey,
    [switch]$PrintCommand
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$resolvedModel = (Resolve-Path -LiteralPath $ModelPath).Path
$resolvedApiKey = (Resolve-Path -LiteralPath $ApiKeyFile).Path
if ((Get-Item -LiteralPath $resolvedApiKey).Length -eq 0) {
    throw "API key file must not be empty"
}

$stateRoot = if ([IO.Path]::IsPathRooted($StateDirectory)) {
    [IO.Path]::GetFullPath($StateDirectory)
} else {
    [IO.Path]::GetFullPath((Join-Path $projectRoot $StateDirectory))
}
$checkpointPath = Join-Path $stateRoot "benefit-$InstanceId.json"

$server = if ($Backend -eq "cuda") {
    Join-Path $projectRoot "build\patched-cuda-ninja3\bin\llama-server.exe"
} else {
    Join-Path $projectRoot "build\patched-cpu-noui\bin\Release\llama-server.exe"
}
if (-not (Test-Path -LiteralPath $server)) {
    throw "server binary missing for backend '$Backend': $server"
}

# Exact model bytes and the serving envelope prevent latency evidence from a
# different model, host, backend, or batching shape from being restored.
if ([string]::IsNullOrWhiteSpace($CheckpointKey)) {
    $modelSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $resolvedModel).Hash
    $machine = [Environment]::MachineName
    $CheckpointKey = "model_sha256=$modelSha256;host=$machine;backend=$Backend;ctx=$ContextSize;parallel=$Parallel"
}

$serverArgs = @(
    "-m", $resolvedModel,
    "--host", "127.0.0.1",
    "--port", "$Port",
    "-c", "$ContextSize",
    "-np", "$Parallel",
    "-t", "$Threads",
    "-ngl", $(if ($Backend -eq "cuda") { "99" } else { "0" }),
    "--api-key-file", $resolvedApiKey,
    "--metrics",
    "--scheduler-policy", "cacheflow",
    "--benefit-policy", "learned",
    "--benefit-checkpoint", $checkpointPath,
    "--benefit-checkpoint-key", $CheckpointKey,
    "--benefit-checkpoint-interval", "128",
    "--kv-block-runtime",
    "--kv-block-size", "16"
)

if ($PrintCommand) {
    [PSCustomObject]@{
        executable = $server
        backend = $Backend
        listen = "127.0.0.1:$Port"
        model = $resolvedModel
        api_key_file = $resolvedApiKey
        checkpoint = $checkpointPath
        checkpoint_key = $CheckpointKey
    } | ConvertTo-Json
    exit 0
}

New-Item -ItemType Directory -Force -Path $stateRoot | Out-Null

if ($Backend -eq "cuda") {
    $cudaBin = Join-Path $projectRoot "runtime\cuda-dev\Library\bin"
    $env:PATH = $cudaBin + [IO.Path]::PathSeparator + $env:PATH
}

& $server @serverArgs
exit $LASTEXITCODE
