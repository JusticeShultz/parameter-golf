param(
    [string]$LogPath = ""
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$project = Join-Path $root "tools\RunMonitor\RunMonitor.csproj"
if (-not (Test-Path $project)) {
    throw "RunMonitor project not found at $project"
}

$arguments = @("run", "--project", $project)
if (-not [string]::IsNullOrWhiteSpace($LogPath)) {
    $arguments += "--"
    $arguments += (Resolve-Path $LogPath).Path
}

& dotnet @arguments
