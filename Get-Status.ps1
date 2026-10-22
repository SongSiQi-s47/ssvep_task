$ErrorActionPreference = 'Stop'
$ssvepRoot = $PSScriptRoot
$ssvepData = @{}
foreach ($ssvepFile in @('download_status.json', 'nakanishi_download_status.json')) {
    $ssvepPath = Join-Path $ssvepRoot ('logs\' + $ssvepFile)
    if (Test-Path -LiteralPath $ssvepPath) {
        $ssvepStatus = Get-Content -LiteralPath $ssvepPath -Raw | ConvertFrom-Json
        foreach ($ssvepProp in $ssvepStatus.PSObject.Properties) {
            if (-not $ssvepData.ContainsKey($ssvepProp.Name) -or $ssvepProp.Value.status -eq 'verified') {
                $ssvepData[$ssvepProp.Name] = $ssvepProp.Value
            }
        }
    }
}
foreach ($ssvepDataset in @('beta', 'nakanishi2015')) {
    $ssvepCount = @($ssvepData.GetEnumerator() | Where-Object { $_.Key.StartsWith($ssvepDataset + '/') -and $_.Key.EndsWith('.mat') -and $_.Value.status -eq 'verified' }).Count
    $ssvepExpected = if ($ssvepDataset -eq 'beta') { 70 } else { 10 }
    Write-Output "$ssvepDataset verified EEG files: $ssvepCount / $ssvepExpected"
}
$ssvepPidPath = Join-Path $ssvepRoot 'logs\download.pid'
if (Test-Path -LiteralPath $ssvepPidPath) {
    $ssvepSavedId = [int](Get-Content -LiteralPath $ssvepPidPath)
    $ssvepWorker = Get-CimInstance Win32_Process -Filter "ProcessId = $ssvepSavedId"
    if ($ssvepWorker -and $ssvepWorker.CommandLine.Contains((Join-Path $ssvepRoot 'scripts\download_public.py'))) {
        Write-Output "Background downloader running. PID: $ssvepSavedId"
    } else {
        Write-Output 'Downloader stopped. Check logs/download_summary.json and logs/download.stderr.log.'
    }
}
