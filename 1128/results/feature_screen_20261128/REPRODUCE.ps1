param(
    [string]$Output = "../../results/feature_screen_20261128_rerun",
    [string]$Python = "D:/download/Anaconda/envs/pytorch/python.exe"
)
$ErrorActionPreference = "Stop"
$CodeDirectory = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "../../code/e1_v04"))
Push-Location -LiteralPath $CodeDirectory
try {
    & $Python verify_screen_rules.py --output (Join-Path $Output "audit/rules_check_reproduced.json")
    if ($LASTEXITCODE -ne 0) { throw "Operating-rule verification failed." }
    & $Python run_feature_screen.py --stage all --jobs 4 --output $Output
    if ($LASTEXITCODE -ne 0) { throw "Development campaign failed; see per-group logs." }
    & $Python analyze_feature_screen.py --output $Output
    if ($LASTEXITCODE -ne 0) { throw "Development summary failed." }
} finally {
    Pop-Location
}
