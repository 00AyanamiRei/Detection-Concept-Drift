param(
    [string]$PythonExe = "e:/bcpraca/study/prototype/project/.venv/Scripts/python.exe",
    [int]$MaxInstances = 20000,
    [int]$Seed = 42,
    [string]$OutputRoot = "experiments/results_wta_elec2_n20000",
    [string]$ReportPath = "experiments/results_wta_elec2_n20000/elec2_parameter_sweep_complete.txt"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$tests = @(
    @{ id = "base"; ws = 300; th = 0.5; al = 1.5 },
    @{ id = "w300"; ws = 300; th = 0.5; al = 1.5 },
    @{ id = "w400"; ws = 400; th = 0.5; al = 1.5 },
    @{ id = "w500"; ws = 500; th = 0.5; al = 1.5 },
    @{ id = "t03"; ws = 300; th = 0.3; al = 1.5 },
    @{ id = "t05"; ws = 300; th = 0.5; al = 1.5 },
    @{ id = "t07"; ws = 300; th = 0.7; al = 1.5 },
    @{ id = "t085"; ws = 300; th = 0.85; al = 1.5 },
    @{ id = "a08"; ws = 300; th = 0.5; al = 0.8 },
    @{ id = "a12"; ws = 300; th = 0.5; al = 1.2 },
    @{ id = "a15"; ws = 300; th = 0.5; al = 1.5 },
    @{ id = "a25"; ws = 300; th = 0.5; al = 2.5 }
)

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

"================ ELEC2 PARAMETER SWEEP COMPLETE (N=$MaxInstances) $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ================" | Set-Content -Path $ReportPath -Encoding UTF8
"" | Add-Content -Path $ReportPath -Encoding UTF8

$failed = @()
$index = 1

foreach ($t in $tests) {
    "------------------------------------------------------------" | Add-Content -Path $ReportPath -Encoding UTF8
    ("TEST {0}/{1}: {2}" -f $index, $tests.Count, $t.id) | Add-Content -Path $ReportPath -Encoding UTF8

    $runId = "n${MaxInstances}_$($t.id)"
    $args = @(
        "main.py",
        "--dataset", "elec2",
        "--max-instances", "$MaxInstances",
        "--seed", "$Seed",
        "--window-size", "$($t.ws)",
        "--theta", "$($t.th)",
        "--alpha", "$($t.al)",
        "--output", "$OutputRoot",
        "--run-id", "$runId"
    )

    ("COMMAND: {0} {1}" -f $PythonExe, ($args -join " ")) | Add-Content -Path $ReportPath -Encoding UTF8
    ("START: {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss")) | Add-Content -Path $ReportPath -Encoding UTF8

    $output = & $PythonExe @args 2>&1 | Out-String -Width 4096
    $output | Add-Content -Path $ReportPath -Encoding UTF8

    ("EXIT_CODE: {0}" -f $LASTEXITCODE) | Add-Content -Path $ReportPath -Encoding UTF8
    ("END: {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss")) | Add-Content -Path $ReportPath -Encoding UTF8

    if ($LASTEXITCODE -ne 0) {
        $failed += $t.id
    }

    $index++
}

"================ END ELEC2 PARAMETER SWEEP COMPLETE (N=$MaxInstances) ================" | Add-Content -Path $ReportPath -Encoding UTF8

if ($failed.Count -gt 0) {
    throw "Sweep finished with failed tests: $($failed -join ', ')"
}

Write-Host "Sweep finished successfully. Consolidated report: $ReportPath"
