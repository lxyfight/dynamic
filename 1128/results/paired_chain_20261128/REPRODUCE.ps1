param(
    [string]$Output = '../../results/paired_chain_rerun',
    [string]$Python = 'D:/download/Anaconda/envs/pytorch/python.exe',
    [int]$Jobs = 12
)
$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'
$CodeDirectory = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../code/e1_v04'))
Push-Location -LiteralPath $CodeDirectory
try {
    foreach ($Split in @('train','tune','dev')) {
        & $Python run_paired_chain_campaign.py --stage capture --split $Split --jobs $Jobs --output $Output
        if ($LASTEXITCODE -ne 0) { throw "Capture failed: $Split" }
        & $Python run_paired_chain_campaign.py --stage evaluate --split $Split --jobs $Jobs --output $Output
        if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $Split" }
    }
    & $Python fit_paired_chain.py --stage base --defer-dev --output $Output
    if ($LASTEXITCODE -ne 0) { throw 'Base fitting failed.' }
    & $Python fit_paired_chain.py --stage augmented --output $Output
    if ($LASTEXITCODE -ne 0) { throw 'Augmented fitting failed.' }
    & $Python fit_paired_chain.py --stage base --output $Output
    if ($LASTEXITCODE -ne 0) { throw 'Base evaluation failed.' }
    & $Python audit_paired_chain_campaign.py --output $Output
    if ($LASTEXITCODE -ne 0) { throw 'Independent audit failed.' }
    & $Python report_paired_chain.py --output $Output
    if ($LASTEXITCODE -ne 0) { throw 'Report generation failed.' }
} finally { Pop-Location }
