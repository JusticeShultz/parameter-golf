param(
    [string]$Only = "",
    [switch]$Background
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$launcher = Join-Path $root "scripts\run_local_3090.ps1"
$summarizer = Join-Path $root "scripts\summarize_log.py"
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $launcher)) {
    throw "Launcher not found at $launcher"
}

if ($Background -and [string]::IsNullOrWhiteSpace($Only)) {
    throw "Background mode requires -Only so we do not start multiple GPU jobs at once."
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$experiments = @(
    @{
        Name = "proxy_qat_s"
        MaxWallclockSeconds = 180
        TrainBatchTokens = 32768
        ValBatchSize = 32768
        ValMaxTokens = 1048576
        RoundtripValMaxTokens = 524288
        ValLossEvery = 25
        NumLayers = 12
        NumUniqueBlocks = 3
        ModelDim = 384
        EmbedDim = 192
        NumHeads = 6
        NumKvHeads = 3
        CompressionRegWeight = 0.01
        TernaryRegWeight = 0.1
        EvalCacheMixWeight = 0.03
    },
    @{
        Name = "proxy_factored_m"
        MaxWallclockSeconds = 240
        TrainBatchTokens = 49152
        ValBatchSize = 49152
        ValMaxTokens = 2097152
        RoundtripValMaxTokens = 1048576
        ValLossEvery = 30
        NumLayers = 14
        NumUniqueBlocks = 4
        ModelDim = 448
        EmbedDim = 224
        NumHeads = 8
        NumKvHeads = 4
        CompressionRegWeight = 0.015
        TernaryRegWeight = 0.12
        EvalCacheMixWeight = 0.04
    },
    @{
        Name = "proxy_qat_l"
        MaxWallclockSeconds = 300
        TrainBatchTokens = 65536
        ValBatchSize = 65536
        ValMaxTokens = 4194304
        RoundtripValMaxTokens = 2097152
        ValLossEvery = 40
        NumLayers = 16
        NumUniqueBlocks = 4
        ModelDim = 512
        EmbedDim = 0
        NumHeads = 8
        NumKvHeads = 4
        CompressionRegWeight = 0.02
        TernaryRegWeight = 0.15
        EvalCacheMixWeight = 0.05
    }
)

if (-not [string]::IsNullOrWhiteSpace($Only)) {
    $experiments = @($experiments | Where-Object { $_.Name -eq $Only })
    if ($experiments.Count -eq 0) {
        throw "No sweep preset matched -Only '$Only'"
    }
}

foreach ($experiment in $experiments) {
    $runId = "{0}_{1}" -f $experiment.Name, $stamp
    $args = @(
        "-ExecutionPolicy", "Bypass",
        "-File", $launcher,
        "-RunId", $runId,
        "-MaxWallclockSeconds", $experiment.MaxWallclockSeconds.ToString(),
        "-TrainBatchTokens", $experiment.TrainBatchTokens.ToString(),
        "-ValBatchSize", $experiment.ValBatchSize.ToString(),
        "-ValMaxTokens", $experiment.ValMaxTokens.ToString(),
        "-RoundtripValMaxTokens", $experiment.RoundtripValMaxTokens.ToString(),
        "-TrainLogEvery", "10",
        "-ValLossEvery", $experiment.ValLossEvery.ToString(),
        "-WarmupSteps", "0",
        "-NumLayers", $experiment.NumLayers.ToString(),
        "-NumUniqueBlocks", $experiment.NumUniqueBlocks.ToString(),
        "-ModelDim", $experiment.ModelDim.ToString(),
        "-EmbedDim", $experiment.EmbedDim.ToString(),
        "-NumHeads", $experiment.NumHeads.ToString(),
        "-NumKvHeads", $experiment.NumKvHeads.ToString(),
        "-MlpMult", "2",
        "-CompressionRegWeight", $experiment.CompressionRegWeight.ToString([System.Globalization.CultureInfo]::InvariantCulture),
        "-CompressionRegInterval", "4",
        "-CompressionRegWarmupSteps", "8",
        "-CompressionRegSampleTensors", "4",
        "-CompressionRegMaxCols", "128",
        "-TernaryRegWeight", $experiment.TernaryRegWeight.ToString([System.Globalization.CultureInfo]::InvariantCulture),
        "-OutlierRegWeight", "0.01",
        "-EvalCacheMixWeight", $experiment.EvalCacheMixWeight.ToString([System.Globalization.CultureInfo]::InvariantCulture),
        "-EvalCacheSize", "8",
        "-SaveRawCheckpoint", "0",
        "-FinalRoundtripEval", "0"
    )

    if ($Background) {
        $args += "-Background"
    }

    Write-Output ("Launching {0}" -f $runId)
    & powershell @args

    if ($Background) {
        break
    }

    $logPath = Join-Path $root ("logs\{0}.txt" -f $runId)
    if ((Test-Path $python) -and (Test-Path $summarizer) -and (Test-Path $logPath)) {
        & $python $summarizer $logPath
    }
}
