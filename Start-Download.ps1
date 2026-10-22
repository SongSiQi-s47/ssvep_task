$ErrorActionPreference = 'Stop'
$ssvepRoot = $PSScriptRoot
$ssvepScript = Join-Path $ssvepRoot 'scripts\download_public.py'
$ssvepPython = 'C:\Users\64443\anaconda3\python.exe'
$ssvepExisting = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object {
    $_.CommandLine -and $_.CommandLine.Contains($ssvepScript)
}
if ($ssvepExisting) {
    Write-Output "Download already running. PID: $($ssvepExisting.ProcessId)"
    exit 0
}
$ssvepProcess = Start-Process -FilePath $ssvepPython -ArgumentList @('-u', ('"' + $ssvepScript + '"'), '--workers', '4') -WorkingDirectory $ssvepRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $ssvepRoot 'logs\download.stdout.log') -RedirectStandardError (Join-Path $ssvepRoot 'logs\download.stderr.log') -PassThru
$ssvepProcess.Id | Set-Content -LiteralPath (Join-Path $ssvepRoot 'logs\download.pid')
Write-Output "Started download. PID: $($ssvepProcess.Id)"
