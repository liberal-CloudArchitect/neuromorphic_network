param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = if ($env:BRAIN_PYTHON) {
    $env:BRAIN_PYTHON
} else {
    "E:\conda\envs\brain\python.exe"
}

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    Write-Error "brain Python not found: $Python"
    exit 2
}

Set-Location -LiteralPath $Root
& $Python "scripts/p4_control.py" @Arguments
exit $LASTEXITCODE
