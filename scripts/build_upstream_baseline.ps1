$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$forkRoot = Join-Path $projectRoot "vendor\llama.cpp"
$sourceRoot = Join-Path $projectRoot "runtime\upstream-src"
$buildRoot = Join-Path $projectRoot "runtime\upstream-build"
$revision = "acd79d603cb2e1c84c0886137b80f1ad649b6857"

if (-not (Test-Path -LiteralPath (Join-Path $sourceRoot ".git"))) {
    & git -C $forkRoot worktree add $sourceRoot $revision
    if ($LASTEXITCODE -ne 0) { throw "failed to create pinned upstream worktree" }
}

& cmake -S $sourceRoot -B $buildRoot -G "Visual Studio 17 2022" -A x64 `
    -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF `
    -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_TOOLS=ON `
    -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF `
    -DLLAMA_UI_GZIP=OFF
if ($LASTEXITCODE -ne 0) { throw "pinned upstream configure failed" }

& cmake --build $buildRoot --config Release --target llama-server -j 8
if ($LASTEXITCODE -ne 0) { throw "pinned upstream build failed" }

$server = Join-Path $buildRoot "bin\Release\llama-server.exe"
if (-not (Test-Path -LiteralPath $server)) { throw "pinned upstream server missing" }
Write-Host "Pinned same-toolchain upstream server ready: $server"
