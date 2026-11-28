param(
    [string]$Output = "../../results/paired_wait_rerun",
    [string]$Python = "D:/download/Anaconda/envs/pytorch/python.exe",
    [int]$Jobs = 12
)
$ErrorActionPreference = "Stop"
$CodeDirectory = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "../../code/e1_v04"))
Push-Location -LiteralPath $CodeDirectory
try {
    & $Python run_paired_wait_diagnostic.py --stage all --samples 8 --jobs $Jobs --output $Output
    if ($LASTEXITCODE -ne 0) { throw "Paired replay failed." }
    & $Python score_paired_wait_chain.py --output $Output
    if ($LASTEXITCODE -ne 0) { throw "Frozen action scoring failed." }
    & $Python analyze_paired_wait_diagnostic.py --output $Output
    if ($LASTEXITCODE -ne 0) { throw "Paired result audit failed." }
} finally {
    Pop-Location
}
