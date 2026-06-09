param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('uad_source', 'adaptnas_combined')]
    [string]$Mode,

    [ValidateSet('default_nasade', 'omni_anomaly', 'usad')]
    [string]$Family = 'default_nasade',
    
    [string]$DataDir = '',
    [string]$RawSmdRoot = 'data/ServerMachineDataset',
    [string]$Machine = 'machine-1-1',
    [string]$SwatTrainCsv = 'data/SWaT/SWaT_Dataset_Normal_v1.csv',
    [string]$SwatTestCsv = 'data/SWaT/SWaT_Dataset_Attack_v0.csv',
    [int]$EpochsPretrain = 50,
    [int]$SearchCandidates = 20,
    [int]$BatchSize = 128,
    [ValidateSet('deepsvdd', 'autoencoder', 'knn_distance', 'oneclass_svm', 'svdd', 'prototype_oneclass', 'mahalanobis_head', 'gmm_head')]
    [string]$OneClassMethod = 'deepsvdd',
    [int]$OneClassEpochs = 10,
    [int]$OneClassFinalEpochs = 20,
    [double]$OneClassLr = 0.001,
    [int]$OneClassBatchSize = 1024,
    [int]$OneClassMaxFit = 5000,
    [int]$KnnK = 5,
    [double]$OcsvmNu = 0.05,
    [ValidateSet('linear', 'rbf', 'poly', 'sigmoid')]
    [string]$OcsvmKernel = 'rbf',
    [string]$OcsvmGamma = 'scale',
    [int]$OcsvmDegree = 3,
    [double]$OcsvmCoef0 = 0.0,
    [int]$SvddHiddenDim = 128,
    [int]$SvddRepDim = 64,
    [double]$SvddNu = 0.05,
    [int]$SvddWarmupEpochs = 2,
    [int]$SvddFinalWarmupEpochs = 5,
    [int]$AeHiddenDim = 128,
    [int]$AeLatentDim = 64,
    [int]$MahaHiddenDim = 128,
    [int]$MahaRepDim = 64,
    [double]$MahaShrinkage = 0.01,
    [int]$GmmHiddenDim = 128,
    [int]$GmmRepDim = 64,
    [int]$GmmComponents = 3,
    [ValidateSet('diag', 'full')]
    [string]$GmmCovarianceType = 'diag',
    [double]$GmmRegCovar = 0.0001,
    [int]$GmmWarmupEpochs = 2,
    [int]$ProtoHiddenDim = 128,
    [int]$ProtoRepDim = 64,
    [int]$ProtoCount = 4,
    [double]$ProtoSeparationWeight = 0.1,
    [double]$ProtoSeparationMargin = 1.0,
    [int]$NasSearchIters = 3,
    [int]$CombinedSearchCandidateWarmupSteps = 50,
    [int]$CombinedSearchSteps = 80,
    [int]$CombinedFinalCandidateWarmupSteps = 80,
    [int]$CombinedFinalSteps = 200,
    [int]$CombinedFinalPatience = 10,
    [int]$OmniEpochs = 20,
    [int]$OmniFinalEpochs = 20,
    [double]$OmniLr = 0.001,
    [int]$OmniPatience = 5,
    [int]$OmniWindowLength = 100,
    [double]$OmniValidRatio = 0.3,
    [int]$OmniBatchSize = 50,
    [int]$OmniStride = 1,
    [int]$OmniTestNZ = 1,
    [int]$OmniSearchIters = 3,
    [int]$OmniTrainLimit = 0,
    [int]$OmniTestLimit = 0,
    [ValidateSet('paper', 'repo')]
    [string]$OmniReference = 'paper',
    [ValidateSet('official_minmax', 'train_zscore')]
    [string]$OmniPreprocess = 'official_minmax',
    [switch]$OmniFixedOnly,
    [double]$OmniPotQ = 0.0,
    [double]$OmniPotLevel = 0.0,
    [int]$UsadEpochs = 70,
    [int]$UsadFinalEpochs = 70,
    [double]$UsadLr = 0.001,
    [int]$UsadPatience = 5,
    [int]$UsadWindowLength = 12,
    [double]$UsadValidRatio = 0.2,
    [int]$UsadBatchSize = 128,
    [int]$UsadStride = 1,
    [int]$UsadDownsample = 5,
    [int]$UsadLatentSize = 40,
    [int]$UsadSearchIters = 3,
    [int]$UsadTrainLimit = 0,
    [int]$UsadTestLimit = 0,
    [double]$UsadScoreAlpha = 0.5,
    [double]$UsadScoreBeta = 0.5,
    [ValidateSet('train_minmax', 'train_zscore')]
    [string]$UsadPreprocess = 'train_minmax',
    [switch]$UsadFixedOnly,
    [double]$UsadPotQ = 0.001,
    [double]$UsadPotLevel = 0.99,
    [ValidateSet('auto', 'cpu', 'cuda')]
    [string]$Device = 'auto'
)

# Define dataset paths for each mode
$DatasetPaths = @{
    'uad_source' = 'data/smd/machine-1-1/train_normal.npz,data/smd/machine-1-1/val_mixed.npz,data/smd/machine-1-1/test_mixed.npz'
    'adaptnas_combined' = 'data/smd/machine-1-1/train_normal.npz,data/smd/machine-1-1/target_pool_unlabeled.npz,data/smd/machine-1-1/val_mixed.npz,data/smd/machine-1-1/test_mixed.npz'
}

function Test-RawSmdLayout {
    param([string]$RootPath)

    if ([string]::IsNullOrWhiteSpace($RootPath)) {
        return $false
    }

    return (
        (Test-Path (Join-Path $RootPath 'train')) -and
        (Test-Path (Join-Path $RootPath 'test')) -and
        (
            (Test-Path (Join-Path $RootPath 'test_label')) -or
            (Test-Path (Join-Path $RootPath 'labels'))
        )
    )
}

function Resolve-RawSmdRoot {
    param([string]$RequestedRoot)

    $projectRoot = (Get-Location).Path
    $candidates = @()

    if (-not [string]::IsNullOrWhiteSpace($RequestedRoot)) {
        $candidates += $RequestedRoot
    }
    $candidates += @(
        'data/ServerMachineDataset',
        'external/OmniAnomaly/ServerMachineDataset',
        'external/tranad_upstream/data/SMD'
    )

    foreach ($candidate in $candidates) {
        $expanded = if ([System.IO.Path]::IsPathRooted($candidate)) {
            $candidate
        }
        else {
            Join-Path $projectRoot $candidate
        }

        if (Test-RawSmdLayout -RootPath $expanded) {
            return (Resolve-Path $expanded).Path
        }
    }

    if (-not [string]::IsNullOrWhiteSpace($RequestedRoot)) {
        return $RequestedRoot
    }

    return (Join-Path $projectRoot 'data/ServerMachineDataset')
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
    '--mode'
    $Mode
    '--family'
    $Family
    '--epochs_pretrain'
    $EpochsPretrain
    '--search_candidates'
    $SearchCandidates
    '--batch_size'
    $BatchSize
    '--oneclass_method'
    $OneClassMethod
    '--oneclass_epochs'
    $OneClassEpochs
    '--oneclass_final_epochs'
    $OneClassFinalEpochs
    '--oneclass_lr'
    $OneClassLr
    '--oneclass_batch_size'
    $OneClassBatchSize
    '--oneclass_max_fit'
    $OneClassMaxFit
    '--knn_k'
    $KnnK
    '--ocsvm_nu'
    $OcsvmNu
    '--ocsvm_kernel'
    $OcsvmKernel
    '--ocsvm_gamma'
    $OcsvmGamma
    '--ocsvm_degree'
    $OcsvmDegree
    '--ocsvm_coef0'
    $OcsvmCoef0
    '--svdd_hidden_dim'
    $SvddHiddenDim
    '--svdd_rep_dim'
    $SvddRepDim
    '--svdd_nu'
    $SvddNu
    '--svdd_warmup_epochs'
    $SvddWarmupEpochs
    '--svdd_final_warmup_epochs'
    $SvddFinalWarmupEpochs
    '--ae_hidden_dim'
    $AeHiddenDim
    '--ae_latent_dim'
    $AeLatentDim
    '--maha_hidden_dim'
    $MahaHiddenDim
    '--maha_rep_dim'
    $MahaRepDim
    '--maha_shrinkage'
    $MahaShrinkage
    '--gmm_hidden_dim'
    $GmmHiddenDim
    '--gmm_rep_dim'
    $GmmRepDim
    '--gmm_components'
    $GmmComponents
    '--gmm_covariance_type'
    $GmmCovarianceType
    '--gmm_reg_covar'
    $GmmRegCovar
    '--gmm_warmup_epochs'
    $GmmWarmupEpochs
    '--proto_hidden_dim'
    $ProtoHiddenDim
    '--proto_rep_dim'
    $ProtoRepDim
    '--proto_count'
    $ProtoCount
    '--proto_separation_weight'
    $ProtoSeparationWeight
    '--proto_separation_margin'
    $ProtoSeparationMargin
    '--nas_search_iters'
    $NasSearchIters
    '--combined_search_candidate_warmup_steps'
    $CombinedSearchCandidateWarmupSteps
    '--combined_search_steps'
    $CombinedSearchSteps
    '--combined_final_candidate_warmup_steps'
    $CombinedFinalCandidateWarmupSteps
    '--combined_final_steps'
    $CombinedFinalSteps
    '--combined_final_patience'
    $CombinedFinalPatience
    '--omni_epochs'
    $OmniEpochs
    '--omni_final_epochs'
    $OmniFinalEpochs
    '--omni_lr'
    $OmniLr
    '--omni_patience'
    $OmniPatience
    '--omni_window_length'
    $OmniWindowLength
    '--omni_valid_ratio'
    $OmniValidRatio
    '--omni_batch_size'
    $OmniBatchSize
    '--omni_stride'
    $OmniStride
    '--omni_test_n_z'
    $OmniTestNZ
    '--omni_search_iters'
    $OmniSearchIters
    '--omni_train_limit'
    $OmniTrainLimit
    '--omni_test_limit'
    $OmniTestLimit
    '--omni_reference'
    $OmniReference
    '--omni_preprocess'
    $OmniPreprocess
    '--omni_pot_q'
    $OmniPotQ
    '--omni_pot_level'
    $OmniPotLevel
    '--usad_epochs'
    $UsadEpochs
    '--usad_final_epochs'
    $UsadFinalEpochs
    '--usad_lr'
    $UsadLr
    '--usad_patience'
    $UsadPatience
    '--usad_window_length'
    $UsadWindowLength
    '--usad_valid_ratio'
    $UsadValidRatio
    '--usad_batch_size'
    $UsadBatchSize
    '--usad_stride'
    $UsadStride
    '--usad_downsample'
    $UsadDownsample
    '--usad_latent_size'
    $UsadLatentSize
    '--usad_search_iters'
    $UsadSearchIters
    '--usad_train_limit'
    $UsadTrainLimit
    '--usad_test_limit'
    $UsadTestLimit
    '--usad_score_alpha'
    $UsadScoreAlpha
    '--usad_score_beta'
    $UsadScoreBeta
    '--usad_preprocess'
    $UsadPreprocess
    '--usad_pot_q'
    $UsadPotQ
    '--usad_pot_level'
    $UsadPotLevel
    '--device'
    $ResolvedDevice
)

if ($Family -eq 'omni_anomaly') {
    $ResolvedRawSmdRoot = Resolve-RawSmdRoot -RequestedRoot $RawSmdRoot
    $Command += @(
        '--raw_smd_root'
        $ResolvedRawSmdRoot
        '--machine'
        $Machine
    )
    if ($OmniFixedOnly) {
        $Command += '--omni_fixed_only'
    }
}
elseif ($Family -eq 'usad') {
    $Command += @(
        '--swat_train_csv'
        $SwatTrainCsv
        '--swat_test_csv'
        $SwatTestCsv
    )
    if ($UsadFixedOnly) {
        $Command += '--usad_fixed_only'
    }
}
else {
    $DatasetPath = if ([string]::IsNullOrWhiteSpace($DataDir)) {
        $DatasetPaths[$Mode]
    }
    else {
        Build-DatasetPathFromDir -DirPath $DataDir -SelectedMode $Mode
    }

    $Command += @(
        '--dataset_or_paths'
        "`"$DatasetPath`""
    )
}

Write-Host "Running pipeline with mode: $Mode" -ForegroundColor Green
Write-Host "Family: $Family" -ForegroundColor Yellow
if ($Family -eq 'omni_anomaly') {
    Write-Host "Raw SMD root: $ResolvedRawSmdRoot" -ForegroundColor Yellow
    Write-Host "Machine: $Machine" -ForegroundColor Yellow
}
elseif ($Family -eq 'usad') {
    Write-Host "SWaT train CSV: $SwatTrainCsv" -ForegroundColor Yellow
    Write-Host "SWaT test CSV: $SwatTestCsv" -ForegroundColor Yellow
}
elseif (-not [string]::IsNullOrWhiteSpace($DataDir)) {
    Write-Host "Data directory: $DataDir" -ForegroundColor Yellow
}
if ($Family -eq 'default_nasade') {
    Write-Host "One-class method: $OneClassMethod" -ForegroundColor Yellow
}
Write-Host "Resolved device: $ResolvedDevice" -ForegroundColor Yellow
Write-Host "Command: $($Command -join ' ')" -ForegroundColor Cyan
Write-Host ""

# Execute the command
& $Command[0] $Command[1..($Command.Length-1)]
