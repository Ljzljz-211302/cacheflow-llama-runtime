param(
    [Parameter(Mandatory = $true)][string]$ApiKeyFile,
    [string]$KnowledgeRoot = "D:\exam\tuimian-monitor\docs\study",
    [string]$Database = "runtime\interview-assistant.db",
    [string]$LlamaUrl = "http://127.0.0.1:8080",
    [ValidateRange(1, 128)][int]$MaxConcurrent = 8,
    [ValidateRange(1, 65535)][int]$Port = 8766
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$resolvedKnowledge = (Resolve-Path -LiteralPath $KnowledgeRoot).Path
$resolvedApiKey = (Resolve-Path -LiteralPath $ApiKeyFile).Path
$databasePath = if ([IO.Path]::IsPathRooted($Database)) {
    [IO.Path]::GetFullPath($Database)
} else {
    [IO.Path]::GetFullPath((Join-Path $projectRoot $Database))
}

$env:PYTHONPATH = Join-Path $projectRoot "src"
Set-Location $projectRoot
python -m interview_assistant `
    --knowledge-root $resolvedKnowledge `
    --db $databasePath `
    --llama-url $LlamaUrl `
    --api-key-file $resolvedApiKey `
    --host 127.0.0.1 `
    --port $Port `
    --max-concurrent $MaxConcurrent
exit $LASTEXITCODE
