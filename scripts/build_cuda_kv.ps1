param([switch]$Sanitize)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$sourceRoot = Join-Path $projectRoot "vendor\llama.cpp"
$buildRoot = Join-Path $projectRoot "build\patched-cuda-ninja"
$toolkitRoot = Join-Path $projectRoot "runtime\cuda-dev\Library"
$nvcc = Join-Path $toolkitRoot "bin\nvcc.exe"
$vcvars = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
$testExe = Join-Path $buildRoot "bin\test-kv-block-cuda.exe"

if (-not (Test-Path -LiteralPath $nvcc)) {
    throw "CUDA nvcc missing at $nvcc; provision the D-drive CUDA 12.6 environment first"
}
if (-not (Test-Path -LiteralPath $vcvars)) {
    throw "Visual Studio 2022 x64 build environment is missing"
}

$sourceCmake = $sourceRoot.Replace("\", "/")
$buildCmake = $buildRoot.Replace("\", "/")
$toolkitCmake = $toolkitRoot.Replace("\", "/")
$nvccCmake = $nvcc.Replace("\", "/")
$configure = @(
    "call `"$vcvars`"",
    "cmake -S $sourceCmake -B $buildCmake -G Ninja",
    "-DGGML_CUDA=ON -DGGML_NATIVE=ON",
    "-DCUDAToolkit_ROOT=$toolkitCmake -DCMAKE_CUDA_COMPILER=$nvccCmake",
    "-DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release",
    "-DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF",
    "-DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON",
    "-DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_UI_GZIP=OFF"
) -join " "
& cmd.exe /d /c $configure
if ($LASTEXITCODE -ne 0) { throw "CUDA configure failed" }

$build = "call `"$vcvars`" && cmake --build $buildCmake --target test-kv-block-cuda -j 4"
& cmd.exe /d /c $build
if ($LASTEXITCODE -ne 0) { throw "CUDA KV target build failed" }

$env:PATH = (Join-Path $toolkitRoot "bin") + ";" + $env:PATH
& $testExe
if ($LASTEXITCODE -ne 0) { throw "CPU/CUDA KV correctness matrix failed" }

if ($Sanitize) {
    $sanitizer = Join-Path $toolkitRoot "compute-sanitizer\compute-sanitizer.exe"
    if (-not (Test-Path -LiteralPath $sanitizer)) {
        throw "Compute Sanitizer is missing"
    }
    & $sanitizer --tool memcheck --error-exitcode 99 $testExe
    if ($LASTEXITCODE -ne 0) {
        throw "Compute Sanitizer failed; on WDDM run EnableDebuggerInterface.bat as Administrator"
    }
}

Write-Host "CUDA KV correctness target passed on sm_89: $testExe"
