param(
    [string]$ProjectRoot = "E:\neuromorphic",
    [string]$EnvironmentPrefix = "E:\conda\envs\brain",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu126"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "conda is not available on PATH"
}
if (-not (Test-Path -LiteralPath $ProjectRoot -PathType Container)) {
    throw "project directory does not exist: $ProjectRoot"
}

$Python = Join-Path $EnvironmentPrefix "python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    conda create --yes --prefix $EnvironmentPrefix python=3.12 pip setuptools wheel
}

& $Python -m pip install --upgrade pip
& $Python -m pip install "torch==2.12.1" --index-url $TorchIndexUrl
& $Python -m pip install --editable "${ProjectRoot}[dev]"
& $Python (Join-Path $ProjectRoot "scripts\check_environment.py") --require cuda

Write-Output (@{
    environment_prefix = $EnvironmentPrefix
    python = $Python
    project_root = $ProjectRoot
    torch_index_url = $TorchIndexUrl
    status = "ready"
} | ConvertTo-Json -Compress)
