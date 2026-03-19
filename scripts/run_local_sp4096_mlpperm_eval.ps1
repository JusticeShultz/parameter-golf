param(
    [string]$RunId = "",
    [switch]$Background
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$launcher = Join-Path $root "scripts\run_local_3090.ps1"
$evalScript = Join-Path $root "scripts\eval_export_mlp_permutation.py"
$dataPath = Join-Path $root "data\datasets\fineweb10B_sp4096_local"
$tokenizerPath = Join-Path $root "data\tokenizers\fineweb_4096_bpe.model"

if (-not (Test-Path $python)) {
    throw "Python venv not found at $python"
}
if (-not (Test-Path $launcher)) {
    throw "Launcher not found at $launcher"
}
if (-not (Test-Path $evalScript)) {
    throw "Eval script not found at $evalScript"
}
if (-not (Test-Path $dataPath)) {
    throw "SP4096 local dataset not found at $dataPath"
}
if (-not (Test-Path $tokenizerPath)) {
    throw "SP4096 tokenizer not found at $tokenizerPath"
}

if ([string]::IsNullOrWhiteSpace($RunId)) {
    $RunId = "sp4096mlpperm_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}

if ($Background) {
    $args = @("-ExecutionPolicy", "Bypass", "-File", $PSCommandPath, "-RunId", $RunId)
    $proc = Start-Process -FilePath "powershell" -ArgumentList $args -WorkingDirectory $root -PassThru
    Write-Output ("RUN_ID={0}" -f $RunId)
    Write-Output ("PID={0}" -f $proc.Id)
    exit 0
}

$trainArgs = @(
    "-ExecutionPolicy", "Bypass",
    "-File", $launcher,
    "-RunId", $RunId,
    "-DataPath", $dataPath,
    "-TokenizerPath", $tokenizerPath,
    "-VocabSize", "4096",
    "-TieEmbeddings", "1",
    "-MaxWallclockSeconds", "0",
    "-Iterations", "300",
    "-TrainBatchTokens", "32768",
    "-ValBatchSize", "32768",
    "-ValMaxTokens", "524288",
    "-RoundtripValMaxTokens", "262144",
    "-TrainLogEvery", "25",
    "-ValLossEvery", "0",
    "-WarmupSteps", "0",
    "-NumLayers", "14",
    "-NumUniqueBlocks", "14",
    "-ModelDim", "560",
    "-EmbedDim", "0",
    "-NumHeads", "8",
    "-NumKvHeads", "4",
    "-MlpMult", "2",
    "-WindowSize", "0",
    "-Int8AxisMode", "auto",
    "-Int8ResidualRank", "1",
    "-Int8ResidualBudgetBytes", "65536",
    "-CompressionRegWeight", "0.005",
    "-CompressionRegInterval", "4",
    "-CompressionRegWarmupSteps", "32",
    "-CompressionRegSampleTensors", "4",
    "-CompressionRegMaxCols", "128",
    "-CompressionGridRegWeight", "0.10",
    "-CompressionScaleRegWeight", "0.0",
    "-CompressionRank1RegWeight", "0.0",
    "-TernaryRegWeight", "0",
    "-OutlierRegWeight", "0",
    "-EvalCacheMixWeight", "0",
    "-EvalBigramMixWeight", "0",
    "-EvalCacheSize", "0",
    "-WeightQuantizationBits", "8",
    "-EmbedQuantizationBits", "8",
    "-SaveRawCheckpoint", "1",
    "-FinalRoundtripEval", "0",
    "-Background"
)

$launchOutput = & powershell @trainArgs
$pidLine = $launchOutput | Where-Object { $_ -like "PID=*" } | Select-Object -First 1
if (-not $pidLine) {
    throw "Failed to capture training PID from launcher output."
}
$trainPid = [int]($pidLine -replace "^PID=", "")
Wait-Process -Id $trainPid

$checkpointPath = Join-Path $root "final_model.pt"
if (-not (Test-Path $checkpointPath)) {
    throw "Expected checkpoint not found at $checkpointPath"
}

$permLogPath = Join-Path $root "logs\$RunId.perm.txt"
& $python $evalScript `
    --checkpoint-path $checkpointPath `
    --data-path $dataPath `
    --tokenizer-path $tokenizerPath `
    --vocab-size 4096 `
    --num-layers 14 `
    --num-unique-blocks 14 `
    --model-dim 560 `
    --embed-dim 0 `
    --num-heads 8 `
    --num-kv-heads 4 `
    --mlp-mult 2 `
    --tie-embeddings 1 `
    --train-seq-len 1024 `
    --eval-seq-len 0 `
    --eval-stride 0 `
    --sw-eval-batch 32 `
    --val-batch-size 32768 `
    --roundtrip-val-max-tokens 262144 `
    --int8-axis-mode auto `
    --int8-residual-rank 1 `
    --int8-residual-budget-bytes 65536 `
    --weight-quantization-bits 8 `
    --embed-quantization-bits 8 `
    --log-path $permLogPath

if ($LASTEXITCODE -ne 0) {
    throw "Export permutation evaluation failed."
}
