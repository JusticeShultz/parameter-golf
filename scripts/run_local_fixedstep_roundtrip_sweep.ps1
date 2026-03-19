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
    $SweepId = "fixedsteprtsweep_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}

$controllerLog = Join-Path $root "logs\$SweepId.controller.txt"
New-Item -ItemType Directory -Force -Path (Join-Path $root "logs") | Out-Null

function Test-RoundtripRunComplete {
    param(
        [string]$LogPath
    )

    if (-not (Test-Path $LogPath)) {
        return $false
    }

    return [bool](Select-String -Path $LogPath -Pattern '^final_int8_zlib_roundtrip_exact ' -Quiet)
}

$experiments = @(
    @{ Suffix = "base_a"; CacheMix = "0.0";  BigramMix = "0.0";  CacheSize = "0" },
    @{ Suffix = "side_a"; CacheMix = "0.02"; BigramMix = "0.03"; CacheSize = "8" },
    @{ Suffix = "base_b"; CacheMix = "0.0";  BigramMix = "0.0";  CacheSize = "0" },
    @{ Suffix = "side_b"; CacheMix = "0.02"; BigramMix = "0.03"; CacheSize = "8" }
)

foreach ($experiment in $experiments) {
    $runId = "{0}_{1}" -f $SweepId, $experiment.Suffix
    $runLog = Join-Path $root "logs\$runId.txt"
    Add-Content -Path $controllerLog -Value ("[{0}] START {1} cache={2} bigram={3} size={4}" -f (Get-Date -Format s), $runId, $experiment.CacheMix, $experiment.BigramMix, $experiment.CacheSize)
    $args = @(
        "-ExecutionPolicy", "Bypass",
        "-File", $launcher,
        "-RunId", $runId,
        "-MaxWallclockSeconds", "0",
        "-Iterations", "300",
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
        "-Int8ResidualRank", "1",
        "-Int8ResidualBudgetBytes", "65536",
        "-CompressionRegWeight", "0.005",
        "-CompressionRegInterval", "4",
        "-CompressionRegWarmupSteps", "32",
        "-CompressionRegSampleTensors", "4",
        "-CompressionRegMaxCols", "128",
        "-TernaryRegWeight", "0",
        "-OutlierRegWeight", "0",
        "-EvalCacheMixWeight", $experiment.CacheMix,
        "-EvalBigramMixWeight", $experiment.BigramMix,
        "-EvalCacheSize", $experiment.CacheSize,
        "-SaveRawCheckpoint", "0",
        "-FinalRoundtripEval", "1"
    )
    & powershell @args
    if ($LASTEXITCODE -ne 0) {
        Add-Content -Path $controllerLog -Value ("[{0}] FAIL {1}" -f (Get-Date -Format s), $runId)
        throw "Sweep run failed: $runId"
    }
    if (-not (Test-RoundtripRunComplete -LogPath $runLog)) {
        Add-Content -Path $controllerLog -Value ("[{0}] FAIL {1} missing_final_roundtrip_metric" -f (Get-Date -Format s), $runId)
        throw "Sweep run missing final roundtrip metric: $runId"
    }
    Add-Content -Path $controllerLog -Value ("[{0}] DONE {1}" -f (Get-Date -Format s), $runId)
}

Add-Content -Path $controllerLog -Value ("[{0}] SWEEP_DONE {1}" -f (Get-Date -Format s), $SweepId)
