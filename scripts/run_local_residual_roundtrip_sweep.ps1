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
    $SweepId = "residualrtsweep_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}

$controllerLog = Join-Path $root "logs\$SweepId.controller.txt"
New-Item -ItemType Directory -Force -Path (Join-Path $root "logs") | Out-Null

$experiments = @(
    @{ Suffix = "r0_b000000"; Rank = "0"; Budget = "0" },
    @{ Suffix = "r1_b065536"; Rank = "1"; Budget = "65536" },
    @{ Suffix = "r1_b262144"; Rank = "1"; Budget = "262144" },
    @{ Suffix = "r1_b524288"; Rank = "1"; Budget = "524288" },
    @{ Suffix = "r1_b1048576"; Rank = "1"; Budget = "1048576" }
)

foreach ($experiment in $experiments) {
    $runId = "{0}_{1}" -f $SweepId, $experiment.Suffix
    Add-Content -Path $controllerLog -Value ("[{0}] START {1} rank={2} residual_budget_bytes={3}" -f (Get-Date -Format s), $runId, $experiment.Rank, $experiment.Budget)
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
        "-Int8AxisMode", "auto",
        "-Int8ResidualRank", $experiment.Rank,
        "-Int8ResidualBudgetBytes", $experiment.Budget,
        "-CompressionRegWeight", "0.005",
        "-CompressionRegInterval", "4",
        "-CompressionRegWarmupSteps", "32",
        "-CompressionRegSampleTensors", "4",
        "-CompressionRegMaxCols", "128",
        "-TernaryRegWeight", "0",
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
