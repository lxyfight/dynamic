param([string]$Python = 'D:/download/Anaconda/envs/pytorch/python.exe')
$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'
$ResultDirectory = $PSScriptRoot
$CodeDirectory = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../code/e1_v04'))
Push-Location -LiteralPath $CodeDirectory
try {
    while (-not ((Test-Path -LiteralPath "$ResultDirectory/train/evaluation_complete.json") -and (Test-Path -LiteralPath "$ResultDirectory/tune/evaluation_complete.json"))) { Start-Sleep -Seconds 5 }
    & $Python fit_paired_chain.py --stage base --defer-dev --output $ResultDirectory *> "$ResultDirectory/fit_base_freeze.log"
    if ($LASTEXITCODE -ne 0) { throw 'Base fitting failed; see fit_base_freeze.log.' }
    while (-not (Test-Path -LiteralPath "$ResultDirectory/dev/evaluation_complete.json")) { Start-Sleep -Seconds 5 }
    & $Python fit_paired_chain.py --stage augmented --output $ResultDirectory *> "$ResultDirectory/fit_augmented.log"
    if ($LASTEXITCODE -ne 0) { throw 'Augmented fitting/evaluation failed; see fit_augmented.log.' }
    & $Python fit_paired_chain.py --stage base --output $ResultDirectory *> "$ResultDirectory/fit_base_final.log"
    if ($LASTEXITCODE -ne 0) { throw 'Base final evaluation failed; see fit_base_final.log.' }
    & $Python audit_paired_chain_campaign.py --output $ResultDirectory *> "$ResultDirectory/audit.log"
    if ($LASTEXITCODE -ne 0) { throw 'Independent paired audit failed; see audit.log.' }
    & $Python -m unittest discover -s tests -p 'test_paired*.py' -v *> "$ResultDirectory/tests.log"
    if ($LASTEXITCODE -ne 0) { throw 'Regression tests failed; see tests.log.' }
    Write-Output 'Both stages and independent audit completed.'
} finally { Pop-Location }
