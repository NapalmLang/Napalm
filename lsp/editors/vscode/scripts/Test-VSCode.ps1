# Runs the actual extension/client in an isolated VS Code profile on Windows.
[CmdletBinding()]
param(
    [string] $CodePath = (Join-Path $env:ProgramFiles 'Microsoft VS Code\Code.exe'),
    [string] $ExtensionPath = (Join-Path $PSScriptRoot '..')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../../..')).Path
$extensionRoot = (Resolve-Path -LiteralPath $ExtensionPath).Path
$testEntry = (Resolve-Path (Join-Path $PSScriptRoot '../tests/vscode-host.cjs')).Path
$codeExecutable = (Resolve-Path -LiteralPath $CodePath).Path
$runRoot = Join-Path $repoRoot ('target/lsp-vscode-' + [Guid]::NewGuid().ToString('N'))
$workspace = Join-Path $runRoot 'workspace'
$profile = Join-Path $runRoot 'profile'
$extensions = Join-Path $runRoot 'extensions'
$report = Join-Path $runRoot 'report.json'
New-Item -ItemType Directory -Path $workspace, $profile, $extensions -Force | Out-Null

$savedElectron = $env:ELECTRON_RUN_AS_NODE
$savedReport = $env:NAPALM_VSCODE_TEST_REPORT
try {
    $env:ELECTRON_RUN_AS_NODE = $null
    $env:NAPALM_VSCODE_TEST_REPORT = $report
    # Start-Process joins arguments on Windows; quote each path to preserve spaces.
    $arguments = @(
        ('--extensionDevelopmentPath="' + $extensionRoot + '"'),
        ('--extensionTestsPath="' + $testEntry + '"'),
        ('--user-data-dir="' + $profile + '"'),
        ('--extensions-dir="' + $extensions + '"'),
        '--disable-extensions', '--disable-updates', '--skip-welcome',
        '--skip-release-notes', '--disable-workspace-trust', ('"' + $workspace + '"')
    )
    $testProcess = Start-Process -FilePath $codeExecutable -ArgumentList $arguments -WindowStyle Hidden -PassThru
    if (-not $testProcess.WaitForExit(60000)) {
        Stop-Process -Id $testProcess.Id -ErrorAction SilentlyContinue
        throw "VS Code tests timed out. Logs: $profile"
    }
    if (-not (Test-Path -LiteralPath $report)) {
        throw "VS Code produced no test report (exit $($testProcess.ExitCode)). Logs: $profile"
    }
    $result = Get-Content -LiteralPath $report -Raw | ConvertFrom-Json
    if ($result.PSObject.Properties['error']) { throw $result.error }
    if ($testProcess.ExitCode -ne 0) { throw "VS Code exited with code $($testProcess.ExitCode)" }
    $result.passed | ForEach-Object { Write-Output "PASS: $_" }
    Write-Output "Report: $report"
} finally {
    $env:ELECTRON_RUN_AS_NODE = $savedElectron
    $env:NAPALM_VSCODE_TEST_REPORT = $savedReport
}
