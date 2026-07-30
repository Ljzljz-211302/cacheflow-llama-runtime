$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$sourceRoot = Join-Path $projectRoot "vendor\llama.cpp"
$buildRoot = Join-Path $projectRoot "build\patched-cpu-noui"
$serverExe = Join-Path $buildRoot "bin\Release\llama-server.exe"
$nativeTests = @(
    "test-inference-scheduler",
    "test-kv-capacity-planner",
    "test-kv-block-manager",
    "test-kv-runtime",
    "test-kv-block-backend",
    "test-speculation-controller"
)

if (-not (Test-Path -LiteralPath (Join-Path $sourceRoot ".git"))) {
    throw "llama.cpp source missing; run scripts\bootstrap.ps1 first"
}

& cmake -S $sourceRoot -B $buildRoot -G "Visual Studio 17 2022" -A x64 `
    -DGGML_CUDA=OFF `
    -DLLAMA_CURL=OFF `
    -DLLAMA_BUILD_TESTS=ON `
    -DLLAMA_BUILD_EXAMPLES=OFF `
    -DLLAMA_BUILD_APP=OFF `
    -DLLAMA_BUILD_TOOLS=ON `
    -DLLAMA_BUILD_SERVER=ON `
    -DLLAMA_BUILD_UI=OFF `
    -DLLAMA_USE_PREBUILT_UI=OFF `
    -DLLAMA_UI_GZIP=OFF
if ($LASTEXITCODE -ne 0) {
    throw "patched server configure failed"
}

& cmake --build $buildRoot --config Release --target llama-server @nativeTests -j 8
if ($LASTEXITCODE -ne 0) {
    throw "patched server build failed"
}

foreach ($test in $nativeTests) {
    $testExe = Join-Path $buildRoot "bin\Release\$test.exe"
    & $testExe
    if ($LASTEXITCODE -ne 0) {
        throw "native test failed: $test"
    }
}

if (-not (Test-Path -LiteralPath $serverExe)) {
    throw "patched llama-server binary missing after build"
}
Write-Host "Patched server ready: $serverExe"
