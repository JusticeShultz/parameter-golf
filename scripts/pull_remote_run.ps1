param(
    [Parameter(Mandatory = $true)]
    [string]$RemoteHost,
    [Parameter(Mandatory = $true)]
    [int]$Port,
    [Parameter(Mandatory = $true)]
    [string]$RunId,
    [string]$User = "root",
    [string]$SshKeyPath = "",
    [string]$RemoteRoot = "/workspace/parameter-golf",
    [string]$LocalOutDir = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($SshKeyPath)) {
    $SshKeyPath = Join-Path $HOME ".ssh\runpod_codex_ed25519"
}
if (-not (Test-Path $SshKeyPath)) {
    throw "SSH key not found at $SshKeyPath"
}

if ([string]::IsNullOrWhiteSpace($LocalOutDir)) {
    $LocalOutDir = Join-Path $repoRoot ("logs\remote\" + $RunId)
}
New-Item -ItemType Directory -Force -Path $LocalOutDir | Out-Null

function Copy-RemoteFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RemotePath,
        [Parameter(Mandatory = $true)]
        [string]$LocalPath
    )
    $checkCmd = "test -f '$RemotePath'"
    & ssh -i $SshKeyPath -p $Port -o StrictHostKeyChecking=accept-new "$User@$RemoteHost" $checkCmd | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Output ("MISSING {0}" -f $RemotePath)
        return
    }
    & scp -P $Port -i $SshKeyPath "$User@${RemoteHost}:$RemotePath" $LocalPath
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to copy $RemotePath"
    }
    Write-Output ("COPIED {0}" -f $RemotePath)
}

$runLogRemote = "$RemoteRoot/logs/$RunId.txt"
$driverLogRemote = "$RemoteRoot/logs/$RunId.driver.log"
$artifactRemote = "$RemoteRoot/final_model.int8.ptz"
$artifactRunScopedRemote = "$RemoteRoot/logs/$RunId.final_model.int8.ptz"
$rawModelRemote = "$RemoteRoot/final_model.pt"

Copy-RemoteFile -RemotePath $runLogRemote -LocalPath (Join-Path $LocalOutDir "$RunId.txt")
Copy-RemoteFile -RemotePath $driverLogRemote -LocalPath (Join-Path $LocalOutDir "$RunId.driver.log")
$artifactOut = Join-Path $LocalOutDir "$RunId.final_model.int8.ptz"
& ssh -i $SshKeyPath -p $Port -o StrictHostKeyChecking=accept-new "$User@$RemoteHost" "test -f '$artifactRunScopedRemote'" | Out-Null
if ($LASTEXITCODE -eq 0) {
    Copy-RemoteFile -RemotePath $artifactRunScopedRemote -LocalPath $artifactOut
}
else {
    Copy-RemoteFile -RemotePath $artifactRemote -LocalPath $artifactOut
}
Copy-RemoteFile -RemotePath $rawModelRemote -LocalPath (Join-Path $LocalOutDir "$RunId.final_model.pt")

Write-Output ("LOCAL_OUT={0}" -f $LocalOutDir)
