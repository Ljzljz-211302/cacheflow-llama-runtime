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
    [ValidatePattern("^[A-Za-z0-9._-]{1,64}$")][string]$CheckpointNamespace = "default",
    [switch]$PrintCommand
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$resolvedModel = (Resolve-Path -LiteralPath $ModelPath).Path
$resolvedApiKey = (Resolve-Path -LiteralPath $ApiKeyFile).Path
if ((Get-Item -LiteralPath $resolvedApiKey).Length -eq 0 -or
        -not (Get-Content -LiteralPath $resolvedApiKey | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })) {
    throw "API key file must contain at least one non-empty key"
}

$stateRoot = if ([IO.Path]::IsPathRooted($StateDirectory)) {
    [IO.Path]::GetFullPath($StateDirectory)
} else {
    [IO.Path]::GetFullPath((Join-Path $projectRoot $StateDirectory))
}
$checkpointPath = Join-Path $stateRoot "benefit-$InstanceId.json"
$lockPath = "$checkpointPath.lock"

$server = if ($Backend -eq "cuda") {
    Join-Path $projectRoot "build\patched-cuda-ninja3\bin\llama-server.exe"
} else {
    Join-Path $projectRoot "build\patched-cpu-noui\bin\Release\llama-server.exe"
}
if (-not (Test-Path -LiteralPath $server)) {
    throw "server binary missing for backend '$Backend': $server"
}

# The caller may partition state with a namespace, but cannot replace the
# compatibility identity. Exact model bytes and the serving envelope are
# always included, so evidence cannot cross models or deployment shapes.
$modelSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $resolvedModel).Hash
$machine = [Environment]::MachineName
$checkpointIdentity = "schema=1;namespace=$CheckpointNamespace;model_sha256=$modelSha256;host=$machine;backend=$Backend;ctx=$ContextSize;parallel=$Parallel"
$sha256 = [Security.Cryptography.SHA256]::Create()
try {
    $identityBytes = [Text.Encoding]::UTF8.GetBytes($checkpointIdentity)
    $identityHash = $sha256.ComputeHash($identityBytes)
    $CheckpointKey = "cacheflow-benefit-v1:" + ([BitConverter]::ToString($identityHash).Replace("-", "").ToLowerInvariant())
} finally {
    $sha256.Dispose()
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
    "--no-webui",
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
        instance_lock = $lockPath
        checkpoint_namespace = $CheckpointNamespace
        checkpoint_key = $CheckpointKey
    } | ConvertTo-Json
    exit 0
}

New-Item -ItemType Directory -Force -Path $stateRoot | Out-Null
$lockStream = $null
try {
    $lockStream = [IO.File]::Open(
        $lockPath,
        [IO.FileMode]::OpenOrCreate,
        [IO.FileAccess]::ReadWrite,
        [IO.FileShare]::None
    )
} catch {
    throw "instance '$InstanceId' already owns checkpoint state or lock is unavailable: $lockPath"
}

try {
    if ($Backend -eq "cuda") {
        $cudaBin = Join-Path $projectRoot "runtime\cuda-dev\Library\bin"
        $env:PATH = $cudaBin + [IO.Path]::PathSeparator + $env:PATH
    }

    & $server @serverArgs
    $serverExitCode = $LASTEXITCODE
} finally {
    $lockStream.Dispose()
}

exit $serverExitCode
