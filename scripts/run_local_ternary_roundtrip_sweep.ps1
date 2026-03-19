param(
    [string]$SweepId = ""
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$launcher = Join-Path $root "scripts\run_local_3090.ps1"
if (-not (Test-Path $launcher)) {
    throw "Launcher not found at $launcher"
}

if ([string]::IsNullOrWhiteSpace($SweepId)) {
    $SweepId = "ternaryrtsweep_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}

$controllerLog = Join-Path $root "logs\$SweepId.controller.txt"
New-Item -ItemType Directory -Force -Path (Join-Path $root "logs") | Out-Null

$experiments = @(
    @{ Suffix = "t0005"; TernaryRegWeight = "0.0005" },
    @{ Suffix = "t0010"; TernaryRegWeight = "0.0010" },
    @{ Suffix = "t0020"; TernaryRegWeight = "0.0020" },
    @{ Suffix = "t0035"; TernaryRegWeight = "0.0035" },
    @{ Suffix = "t0050"; TernaryRegWeight = "0.0050" }
)

foreach ($experiment in $experiments) {
    $runId = "{0}_{1}" -f $SweepId, $experiment.Suffix
    Add-Content -Path $controllerLog -Value ("[{0}] START {1} ternary={2}" -f (Get-Date -Format s), $runId, $experiment.TernaryRegWeight)
    $args = @(
        "-ExecutionPolicy", "Bypass",
        "-File", $launcher,
        "-RunId", $runId,
        "-MaxWallclockSeconds", "180",
        "-TrainBatchTokens", "32768",
        "-ValBatchSize", "32768",
        "-ValMaxTokens", "524288",
        "-RoundtripValMaxTokens", "262144",
        "-TrainLogEvery", "10",
        "-ValLossEvery", "50",
        "-WarmupSteps", "0",
        "-NumLayers", "12",
        "-NumUniqueBlocks", "12",
        "-ModelDim", "384",
        "-EmbedDim", "0",
        "-NumHeads", "6",
        "-NumKvHeads", "3",
        "-MlpMult", "2",
        "-WindowSize", "0",
        "-CompressionRegWeight", "0.005",
        "-CompressionRegInterval", "4",
        "-CompressionRegWarmupSteps", "32",
        "-CompressionRegSampleTensors", "4",
        "-CompressionRegMaxCols", "128",
        "-TernaryRegWeight", $experiment.TernaryRegWeight,
        "-OutlierRegWeight", "0",
        "-EvalCacheMixWeight", "0",
        "-EvalBigramMixWeight", "0",
        "-EvalCacheSize", "0",
        "-SaveRawCheckpoint", "0",
        "-FinalRoundtripEval", "1"
    )
    & powershell @args
    if ($LASTEXITCODE -ne 0) {
        Add-Content -Path $controllerLog -Value ("[{0}] FAIL {1}" -f (Get-Date -Format s), $runId)
        throw "Sweep run failed: $runId"
    }
    Add-Content -Path $controllerLog -Value ("[{0}] DONE {1}" -f (Get-Date -Format s), $runId)
}

Add-Content -Path $controllerLog -Value ("[{0}] SWEEP_DONE {1}" -f (Get-Date -Format s), $SweepId)
