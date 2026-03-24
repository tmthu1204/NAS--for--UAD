param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('uad_source', 'adaptnas_combined')]
    [string]$Mode,
    
    [string]$DataDir = '',
    [int]$EpochsPretrain = 50,
    [int]$SearchCandidates = 20,
    [int]$BatchSize = 128,
    [ValidateSet('auto', 'cpu', 'cuda')]
    [string]$Device = 'auto'
)

# Define dataset paths for each mode
$DatasetPaths = @{
    'uad_source' = 'data/smd/machine-1-1/train_normal.npz,data/smd/machine-1-1/val_mixed.npz,data/smd/machine-1-1/test_mixed.npz'
    'adaptnas_combined' = 'data/smd/machine-1-1/train_normal.npz,data/smd/machine-1-1/target_pool_unlabeled.npz,data/smd/machine-1-1/val_mixed.npz,data/smd/machine-1-1/test_mixed.npz'
}

function Build-DatasetPathFromDir {
    param(
        [string]$DirPath,
        [string]$SelectedMode
    )

    $trainPath = Join-Path $DirPath 'train_normal.npz'
    $valPath = Join-Path $DirPath 'val_mixed.npz'
    $testPath = Join-Path $DirPath 'test_mixed.npz'
    $targetPoolPath = Join-Path $DirPath 'target_pool_unlabeled.npz'

    if (-not (Test-Path $trainPath)) {
        throw "Missing required file: $trainPath"
    }
    if (-not (Test-Path $valPath)) {
        throw "Missing required file: $valPath"
    }

    if ($SelectedMode -eq 'uad_source') {
        $parts = @($trainPath, $valPath)
        if (Test-Path $testPath) {
            $parts += $testPath
        }
        return ($parts -join ',')
    }

    if (-not (Test-Path $targetPoolPath)) {
        throw "Missing required file for adaptnas_combined: $targetPoolPath"
    }

    $parts = @($trainPath, $targetPoolPath, $valPath)
    if (Test-Path $testPath) {
        $parts += $testPath
    }
    return ($parts -join ',')
}

# Get the dataset path for the selected mode
$DatasetPath = if ([string]::IsNullOrWhiteSpace($DataDir)) {
    $DatasetPaths[$Mode]
}
else {
    Build-DatasetPathFromDir -DirPath $DataDir -SelectedMode $Mode
}

$PythonExe = '.\venv\Scripts\python.exe'
if (-not (Test-Path $PythonExe)) {
    throw "Python environment not found at $PythonExe"
}

function Test-CudaAvailable {
    param([string]$PythonPath)

    try {
        $result = & $PythonPath -c "import torch; print('1' if torch.cuda.is_available() else '0')" 2>$null
        return (($result | Select-Object -Last 1).Trim() -eq '1')
    }
    catch {
        return $false
    }
}

$CudaAvailable = Test-CudaAvailable -PythonPath $PythonExe
$ResolvedDevice = $Device

if ($Device -eq 'auto') {
    $ResolvedDevice = if ($CudaAvailable) { 'cuda' } else { 'cpu' }
}
elseif ($Device -eq 'cuda' -and -not $CudaAvailable) {
    Write-Warning "CUDA was requested but this Python environment does not support CUDA. Falling back to CPU."
    $ResolvedDevice = 'cpu'
}

# Build the command
$Command = @(
    $PythonExe
    '-m'
    'src.pipeline'
    '--dataset_or_paths'
    "`"$DatasetPath`""
    '--mode'
    $Mode
    '--epochs_pretrain'
    $EpochsPretrain
    '--search_candidates'
    $SearchCandidates
    '--batch_size'
    $BatchSize
    '--device'
    $ResolvedDevice
)

Write-Host "Running pipeline with mode: $Mode" -ForegroundColor Green
if (-not [string]::IsNullOrWhiteSpace($DataDir)) {
    Write-Host "Data directory: $DataDir" -ForegroundColor Yellow
}
Write-Host "Resolved device: $ResolvedDevice" -ForegroundColor Yellow
Write-Host "Command: $($Command -join ' ')" -ForegroundColor Cyan
Write-Host ""

# Execute the command
& $Command[0] $Command[1..($Command.Length-1)]
