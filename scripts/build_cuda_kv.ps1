param([switch]$Sanitize)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$sourceRoot = Join-Path $projectRoot "vendor\llama.cpp"
$buildRoot = Join-Path $projectRoot "build\patched-cuda-ninja3"
$toolkitRoot = Join-Path $projectRoot "runtime\cuda-dev\Library"
$nvcc = Join-Path $toolkitRoot "bin\nvcc.exe"
$vcvars = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
$testExe = Join-Path $buildRoot "bin\test-kv-block-cuda.exe"
$remapTestExe = Join-Path $buildRoot "bin\test-kv-remap-cuda.exe"
$pagedTestExe = Join-Path $buildRoot "bin\test-paged-decode-cuda.exe"
$backendOpsExe = Join-Path $buildRoot "bin\test-backend-ops.exe"
$policyTestExe = Join-Path $buildRoot "bin\test-kv-action-policy.exe"
$layoutTestExe = Join-Path $buildRoot "bin\test-paged-decode-layout.exe"
$realSwapExe = Join-Path $buildRoot "bin\test-kv-real-cuda-swap.exe"
$model = Join-Path $projectRoot "models\qwen2.5-0.5b-instruct-q4_k_m.gguf"

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
    "call `"$vcvars`" &&",
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

$build = "call `"$vcvars`" && cmake --build $buildCmake --target test-kv-block-cuda test-kv-remap-cuda test-paged-decode-cuda test-kv-real-cuda-swap test-backend-ops test-kv-action-policy test-paged-decode-layout bench-kv-block-cuda bench-kv-cow-cuda bench-paged-decode-cuda llama-server -j 4"
& cmd.exe /d /c $build
if ($LASTEXITCODE -ne 0) { throw "CUDA KV target build failed" }

$env:PATH = (Join-Path $toolkitRoot "bin") + ";" + $env:PATH
& $testExe
if ($LASTEXITCODE -ne 0) { throw "CPU/CUDA KV correctness matrix failed" }
& $remapTestExe
if ($LASTEXITCODE -ne 0) { throw "vectorized CUDA KV remap correctness matrix failed" }
& $pagedTestExe
if ($LASTEXITCODE -ne 0) { throw "restricted paged decode differential tests failed" }
foreach ($variant in @("K1", "K2", "K3", "K4", "K5")) {
    $env:LLAMA_CACHEFLOW_PAGED_KERNEL = $variant
    & $backendOpsExe -b CUDA0 -o FLASH_ATTN_EXT -p "hsk=64,nh=14,nkv=2,kv=64"
    if ($LASTEXITCODE -ne 0) {
        throw "production paged Flash Attention $variant backend-op differential test failed"
    }
}
Remove-Item Env:LLAMA_CACHEFLOW_PAGED_KERNEL -ErrorAction SilentlyContinue
& $policyTestExe
if ($LASTEXITCODE -ne 0) { throw "unified KV action policy tests failed" }
& $layoutTestExe
if ($LASTEXITCODE -ne 0) { throw "production Paged block-table layout tests failed" }
& $realSwapExe $model
if ($LASTEXITCODE -ne 0) { throw "real llama CUDA KV swap test failed" }

if ($Sanitize) {
    $sanitizer = Join-Path $toolkitRoot "compute-sanitizer\compute-sanitizer.exe"
    if (-not (Test-Path -LiteralPath $sanitizer)) {
        throw "Compute Sanitizer is missing"
    }
    & $sanitizer --tool memcheck --error-exitcode 99 $testExe
    if ($LASTEXITCODE -ne 0) {
        throw "Compute Sanitizer memcheck failed; on WDDM run EnableDebuggerInterface.bat as Administrator"
    }
    & $sanitizer --tool racecheck --error-exitcode 99 $testExe
    if ($LASTEXITCODE -ne 0) {
        throw "Compute Sanitizer racecheck failed"
    }
    & $sanitizer --tool memcheck --error-exitcode 99 $remapTestExe
    if ($LASTEXITCODE -ne 0) {
        throw "vectorized KV remap Compute Sanitizer memcheck failed"
    }
    & $sanitizer --tool racecheck --error-exitcode 99 $remapTestExe
    if ($LASTEXITCODE -ne 0) {
        throw "vectorized KV remap Compute Sanitizer racecheck failed"
    }
    & $sanitizer --tool memcheck --error-exitcode 99 $backendOpsExe -b CUDA0 -o FLASH_ATTN_EXT -p "batch=8"
    if ($LASTEXITCODE -ne 0) {
        throw "batched production Paged Flash Attention Compute Sanitizer memcheck failed"
    }
    & $sanitizer --tool racecheck --error-exitcode 99 $backendOpsExe -b CUDA0 -o FLASH_ATTN_EXT -p "batch=4"
    if ($LASTEXITCODE -ne 0) {
        throw "batched production Paged Flash Attention Compute Sanitizer racecheck failed"
    }
}

Write-Host "CUDA KV correctness (including Paged batch 1/2/4/8), real tensor swap, benchmarks, and server targets passed on sm_89."
