param(
    [string]$RunId = "",
    [switch]$Background
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$launcher = Join-Path $root "scripts\run_local_3090.ps1"
if (-not (Test-Path $launcher)) {
    throw "Launcher not found at $launcher"
}

$args = @(
    "-ExecutionPolicy", "Bypass",
    "-File", $launcher,
    "-RunId", $(if ([string]::IsNullOrWhiteSpace($RunId)) { "smoke3090_" + (Get-Date -Format "yyyyMMdd_HHmmss") } else { $RunId }),
    "-MaxWallclockSeconds", "180",
    "-TrainBatchTokens", "32768",
    "-ValBatchSize", "32768",
    "-ValMaxTokens", "1048576",
    "-RoundtripValMaxTokens", "524288",
    "-TrainLogEvery", "10",
    "-ValLossEvery", "25",
    "-WarmupSteps", "0",
    "-NumLayers", "12",
    "-NumUniqueBlocks", "3",
    "-ModelDim", "384",
    "-EmbedDim", "192",
    "-NumHeads", "6",
    "-NumKvHeads", "3",
    "-MlpMult", "2",
    "-CompressionRegWeight", "0.01",
    "-CompressionRegInterval", "4",
    "-CompressionRegWarmupSteps", "8",
    "-CompressionRegSampleTensors", "3",
    "-CompressionRegMaxCols", "128",
    "-TernaryRegWeight", "0.1",
    "-OutlierRegWeight", "0.01",
    "-EvalCacheMixWeight", "0.03",
    "-EvalCacheSize", "8",
    "-SaveRawCheckpoint", "0",
    "-FinalRoundtripEval", "0"
)

if ($Background) {
    $args += "-Background"
}

& powershell @args
