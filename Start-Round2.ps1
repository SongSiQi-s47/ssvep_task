$ErrorActionPreference = 'Stop'
$ssvepScript = Join-Path $PSScriptRoot 'scripts\run_round2_pipeline.py'
$ssvepConfig = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'configs\experiment_v2.json') -Raw | ConvertFrom-Json
$ssvepName = if ($ssvepConfig.training_source -eq 'beta') { 'v2-beta-algorithm-distilled' } else { 'v2-independent-benchmark' }
$ssvepExisting = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object {
    $_.CommandLine -and ($_.CommandLine.Contains($ssvepScript) -or ($_.CommandLine.Contains('ssvep.beta_algorithm_distill') -and $_.CommandLine.Contains($ssvepName)) -or ($_.CommandLine.Contains('ssvep.round2') -and $_.CommandLine.Contains('v2-beta-subject-isolated')))
}
if ($ssvepExisting) { Write-Output "Round 2 already running. PID: $($ssvepExisting.ProcessId)"; exit 0 }
$ssvepRun = Join-Path $PSScriptRoot ('runs\' + $ssvepName + '\completed.json')
if (Test-Path -LiteralPath $ssvepRun) { Write-Output "Round 2 already completed. Inspect runs/$ssvepName."; exit 0 }
$ssvepPython = Join-Path $PSScriptRoot '.venv_gpu\Scripts\python.exe'
$ssvepProcess = Start-Process -FilePath $ssvepPython -ArgumentList @('-u', ('"' + $ssvepScript + '"')) -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $PSScriptRoot 'logs\round2.stdout.log') -RedirectStandardError (Join-Path $PSScriptRoot 'logs\round2.stderr.log') -PassThru
$ssvepProcess.Id | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'logs\round2.pid')
Write-Output "Round 2 started. PID: $($ssvepProcess.Id)"
