# 激活本项目Conda环境；只改变当前PowerShell会话，不修改用户配置或base依赖。
param([string]$CondaRoot = 'D:\Miniconda')

$projectCondaPython = Join-Path $CondaRoot 'python.exe'
if (-not (Test-Path -LiteralPath $projectCondaPython -PathType Leaf)) {
    throw '未找到Conda的Python；请用-CondaRoot指定本机Miniconda目录。'
}
$projectActivation = & $projectCondaPython -B -m conda shell.powershell activate zhixun_env
if ($LASTEXITCODE -ne 0) { throw 'Conda激活失败，请先创建engineering/configs/environment/environment.yml声明的环境。' }
Invoke-Expression ($projectActivation | Out-String)

# 当前机器的exe包装入口不能运行，使用同一Conda的Python模块调用入口。
$Env:CONDA_EXE = $projectCondaPython
$Env:_CONDA_EXE = $projectCondaPython
$Env:_CE_M = '-m'
$Env:_CE_CONDA = 'conda'
Import-Module (Join-Path $CondaRoot 'shell/condabin/Conda.psm1') -Global -ArgumentList @{ChangePs1 = $true}
Write-Output '已激活Conda环境zhixun_env。'
