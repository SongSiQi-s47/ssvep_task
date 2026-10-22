param([string]$RunName = ('experiment-' + (Get-Date -Format 'yyyyMMdd-HHmmss')))
$ErrorActionPreference = 'Stop'
$ssvepPython = Join-Path $PSScriptRoot '.venv_gpu\Scripts\python.exe'
Push-Location $PSScriptRoot
try { & $ssvepPython -m ssvep.experiment --run $RunName; if ($LASTEXITCODE -ne 0) { throw 'Experiment failed; review the error above.' } }
finally { Pop-Location }
