# Essaie attach-nvidia-windows.ps1 sans réseau ni GPU, sur des archives fabriquées (C28-12) : la liste blanche
# « aucun fichier de NVIDIA dans nos archives » est essayée pour Windows comme check-engine-archive.sh l'est pour
# Linux. Chaque cas refusé doit l'être pour sa raison, pas parce que le script plante.
#
#   pwsh tools/release/test-windows-scripts.ps1
$ErrorActionPreference = "Stop"
$script = Join-Path $PSScriptRoot "attach-nvidia-windows.ps1"
$work = Join-Path ([System.IO.Path]::GetTempPath()) "ajp-win-scripts-$([guid]::NewGuid())"
New-Item -ItemType Directory -Path $work | Out-Null
$fail = $false
$env:AJP_RELEASE_REPO = "o/r"

# Un dossier de paquet : l'archive du moteur avec les fichiers donnés, et l'engines.lock qui la décrit.
function New-Package([string]$Name, [string[]]$Files) {
  $dir = Join-Path $work $Name
  $content = Join-Path $dir "contenu"
  New-Item -ItemType Directory -Path $content -Force | Out-Null
  foreach ($file in $Files) { Set-Content -Path (Join-Path $content $file) -Value "x" }
  $archive = Join-Path $dir "ajp-engine-tensorrt-windows-x64.tar.gz"
  tar -C $content -czf $archive .
  if ($LASTEXITCODE -ne 0) { throw "tar a échoué pour $Name" }
  $sha = (Get-FileHash $archive -Algorithm SHA256).Hash.ToLower()
  $size = (Get-Item $archive).Length
  $url = "https://github.com/o/r/releases/download/v1.0.0/ajp-engine-tensorrt-windows-x64.tar.gz"
  Set-Content -Path (Join-Path $dir "engines.lock") -Value @(
    "version = 1", "", "[[engine]]", 'id = "tensorrt"', 'platform = "windows"', "url = `"$url`"", "sha256 = `"$sha`"",
    "size = $size", "", "[[vendor_file]]", 'id = "tensorrt"', 'platform = "windows"',
    'url = "https://pypi.nvidia.com/x"', 'name = "n"')
  return $dir
}

# $Want : 0 attendu réussi, 1 attendu refusé avec $Message dans la sortie.
function Expect([int]$Want, [string]$What, [string]$Message, [string]$Dir) {
  $output = (& pwsh -NoProfile -File $script $Dir "v1.0.0" 2>&1 | Out-String)
  $got = if ($LASTEXITCODE -eq 0) { 0 } else { 1 }
  if ($got -ne $Want) {
    Write-Host "ÉCHEC : $What (code attendu $Want)"; Write-Host $output; $script:fail = $true
  } elseif ($Want -eq 1 -and -not $output.Contains($Message)) {
    Write-Host "ÉCHEC : $What : refusé, mais pas pour « $Message »"; Write-Host $output; $script:fail = $true
  } else { Write-Host "ok : $What" }
}

try {
  Expect 0 "archive du moteur : module et trtexec.exe seulement" "" (New-Package "sain" @("ajp_backend_tensorrt.dll", "trtexec.exe"))
  Expect 1 "archive du moteur : une bibliothèque de NVIDIA en plus est refusée" "autre chose que notre module" `
    (New-Package "intrus" @("ajp_backend_tensorrt.dll", "trtexec.exe", "nvinfer_10.dll"))
  Expect 1 "archive du moteur : trtexec.exe absent est refusé" "trtexec.exe absent" (New-Package "sans-trtexec" @("ajp_backend_tensorrt.dll"))
  Expect 1 "archive du moteur : module absent est refusé" "ajp_backend_tensorrt.dll absent" (New-Package "sans-module" @("trtexec.exe"))
} finally {
  Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
}
if ($fail) { exit 1 }
