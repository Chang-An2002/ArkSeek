param(
    [switch]$Live,
    [switch]$Demo,
    [switch]$NoBrowser,
    [ValidateRange(1, 65535)][int]$Port = 8766
)
$ErrorActionPreference = 'Stop'
if ($Live -and $Demo) { throw '请选择 -Live 或 -Demo，不能同时使用。' }
$adminRoot = $PSScriptRoot
$adminPython = Join-Path $adminRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $adminPython)) { throw '缺少项目 .venv，请先按 README 恢复项目环境。' }
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw "端口 $Port 已被占用；请使用已有后台或用 -Port 选择其他端口。"
}
$adminLogDir = Join-Path $adminRoot 'data\admin_logs'
New-Item -ItemType Directory -Path $adminLogDir -Force | Out-Null
$adminStamp = Get-Date -Format 'yyyyMMdd_HHmmss_ffff'
$adminOut = Join-Path $adminLogDir "admin_$adminStamp.out.log"
$adminErr = Join-Path $adminLogDir "admin_$adminStamp.err.log"
$adminMode = if ($Live) { '--live' } else { '--demo' }
$adminProcess = Start-Process -FilePath $adminPython -ArgumentList @('-u', '-m', 'admin', $adminMode, '--port', $Port) -WorkingDirectory $adminRoot -WindowStyle Hidden -RedirectStandardOutput $adminOut -RedirectStandardError $adminErr -PassThru
$adminUrl = "http://127.0.0.1:$Port"
$adminReady = $false
for ($adminAttempt = 0; $adminAttempt -lt 12; $adminAttempt++) {
    if ($adminProcess.HasExited) { throw "后台启动失败，请查看 $adminErr" }
    try {
        $adminCheck = Invoke-RestMethod -Uri "$adminUrl/api/overview" -TimeoutSec 2
        if ($adminCheck.mode -eq $(if ($Live) { 'live' } else { 'demo' })) { $adminReady = $true; break }
    } catch { Start-Sleep -Milliseconds 300 }
}
if (-not $adminReady) { throw "后台暂未响应，进程号 $($adminProcess.Id)，请查看 $adminErr" }
Write-Output "ArkSeek 后台已启动：$adminUrl"
Write-Output "模式：$(if ($Live) { '本机真实数据（只读）' } else { '虚构演示数据' })"
Write-Output "后台进程号：$($adminProcess.Id)；关闭后台：Stop-Process -Id $($adminProcess.Id)"
if (-not $NoBrowser) { Start-Process $adminUrl }
