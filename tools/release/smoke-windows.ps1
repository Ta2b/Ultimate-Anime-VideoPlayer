# Lance le paquet Windows tel que l'utilisateur le recevra, avant qu'il soit joint puis signé (B51-04) : le zip
# extrait dans un dossier propre, kirari.exe répond (--help : une DLL manquante l'empêche de démarrer) ; puis
# l'installeur, installé en silence dans un dossier neuf, l'kirari.exe installé répond, et la désinstallation retire le
# dossier. Pas d'instantané : le runner n'a pas de Vulkan logiciel.
#
#   pwsh tools/release/smoke-windows.ps1 <kirari-windows-x64.zip> [kirari-windows-x64-setup.exe]
param(
  [Parameter(Mandatory = $true)][string]$Zip,
  [string]$Setup
)
$ErrorActionPreference = "Stop"
$work = Join-Path ([IO.Path]::GetTempPath()) "kirari-smoke-$([guid]::NewGuid().ToString('N'))"
New-Item -ItemType Directory -Force -Path $work | Out-Null

function Test-Start([string]$Exe, [string]$What) {
  if (-not (Test-Path $Exe)) { throw "$What : kirari.exe absent ($Exe)" }
  & $Exe --help | Out-Null
  if ($LASTEXITCODE -ne 0) { throw "$What : kirari.exe --help a rendu le code $LASTEXITCODE (DLL manquante ?)" }
  Write-Host "$What : kirari.exe --help ok"
}

try {
  $portable = Join-Path $work "zip"
  Expand-Archive -Path $Zip -DestinationPath $portable
  $exe = Get-ChildItem $portable -Recurse -Filter kirari.exe | Select-Object -First 1
  if (-not $exe) { throw "$Zip : pas de kirari.exe" }
  Test-Start $exe.FullName "zip"

  if ($Setup) {
    $dir = Join-Path $work "installed"
    $log = Join-Path $work "setup.log"
    $p = Start-Process -FilePath $Setup -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CURRENTUSER", "/DIR=`"$dir`"", "/LOG=`"$log`"" -Wait -PassThru
    if ($p.ExitCode -ne 0) { throw "installeur : code $($p.ExitCode) (journal : $log)" }
    Test-Start (Join-Path $dir "kirari.exe") "installeur"
    $uninstall = Get-ChildItem $dir -Filter "unins*.exe" | Select-Object -First 1
    if (-not $uninstall) { throw "installeur : pas de désinstalleur dans $dir" }
    $p = Start-Process -FilePath $uninstall.FullName -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART" -Wait -PassThru
    if ($p.ExitCode -ne 0) { throw "désinstallation : code $($p.ExitCode)" }
    # Le désinstalleur se retire lui-même un instant après sa sortie.
    for ($i = 0; $i -lt 20 -and (Test-Path (Join-Path $dir "kirari.exe")); $i++) { Start-Sleep -Milliseconds 500 }
    if (Test-Path (Join-Path $dir "kirari.exe")) { throw "désinstallation : kirari.exe est resté dans $dir" }
    Write-Host "installeur : installé, lancé, désinstallé"
  }
} finally {
  Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
}
