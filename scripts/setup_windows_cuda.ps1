param(
    [string]$ProjectRoot = "E:\neuromorphic",
    [string]$EnvironmentPrefix = "E:\conda\envs\brain",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu126",
    [string]$CondaCommand = ""
)

$ErrorActionPreference = "Stop"

if (-not $CondaCommand) {
    $Conda = Get-Command conda -ErrorAction SilentlyContinue
    if ($Conda) {
        $CondaCommand = $Conda.Source
    } else {
        $Candidates = @(
            "E:\Miniconda3\condabin\conda.bat",
            "E:\Miniconda3\Scripts\conda.exe",
            (Join-Path $env:USERPROFILE "miniconda3\condabin\conda.bat"),
            (Join-Path $env:USERPROFILE "anaconda3\condabin\conda.bat")
        )
        $CondaCommand = $Candidates | Where-Object {
            Test-Path -LiteralPath $_ -PathType Leaf
        } | Select-Object -First 1
    }
}
if (-not $CondaCommand -or -not (Test-Path -LiteralPath $CondaCommand -PathType Leaf)) {
    throw "conda is not available on PATH or at a supported installation path"
}
if (-not (Test-Path -LiteralPath $ProjectRoot -PathType Container)) {
    throw "project directory does not exist: $ProjectRoot"
}

$Python = Join-Path $EnvironmentPrefix "python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    & $CondaCommand create --yes --override-channels --channel conda-forge `
        --prefix $EnvironmentPrefix python=3.12 pip setuptools wheel git
    if ($LASTEXITCODE -ne 0) {
        throw "conda environment creation failed with exit code $LASTEXITCODE"
    }
}

& $Python -m pip install --upgrade pip
& $Python -m pip install "torch==2.12.1" --index-url $TorchIndexUrl
& $Python -m pip install --editable "${ProjectRoot}[dev]"
& $Python (Join-Path $ProjectRoot "scripts\check_environment.py") --require cuda

Write-Output (@{
    environment_prefix = $EnvironmentPrefix
    conda = $CondaCommand
    python = $Python
    project_root = $ProjectRoot
    torch_index_url = $TorchIndexUrl
    status = "ready"
} | ConvertTo-Json -Compress)
