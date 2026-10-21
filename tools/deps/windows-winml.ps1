# Le paquet NuGet de Windows ML (Microsoft.Windows.AI.MachineLearning) que construit
# src/gpu/winml (MIGraphX de Windows ML), à la version et à l'empreinte de
# tools/release/versions.env : téléchargé, vérifié, extrait dans le dossier donné. Sous GitHub
# Actions, AJP_WINML_ROOT passe aux étapes suivantes.
#
#   pwsh tools/deps/windows-winml.ps1 C:\deps\winml
param([Parameter(Mandatory = $true)][string]$Dir)
$ErrorActionPreference = "Stop"

$Root = (Resolve-Path "$PSScriptRoot/../..").Path
$Pins = Get-Content "$Root/tools/release/versions.env"
function Pinned([string]$Name) {
  $line = $Pins | Select-String -Pattern "^$Name=(\S+)$"
  if (-not $line) { throw "$Name absent de tools/release/versions.env" }
  return $line.Matches[0].Groups[1].Value
}
$Version = Pinned "WINML_VERSION"
$Sha = Pinned "WINML_SHA256"

New-Item -ItemType Directory -Force -Path $Dir | Out-Null
$Dir = (Resolve-Path $Dir).Path
$Package = Join-Path $Dir "microsoft.windows.ai.machinelearning.$Version.nupkg"
$Url = "https://api.nuget.org/v3-flatcontainer/microsoft.windows.ai.machinelearning/$Version/microsoft.windows.ai.machinelearning.$Version.nupkg"
if (-not (Test-Path $Package)) {
  Invoke-WebRequest -Uri $Url -OutFile $Package -UseBasicParsing
}
$Got = (Get-FileHash $Package -Algorithm SHA256).Hash.ToLower()
if ($Got -ne $Sha) {
  Remove-Item $Package
  throw "Windows ML $Version : empreinte $Got, attendue $Sha"
}
# Un .nupkg est un zip ; tar (bsdtar de Windows) l'ouvre sans l'extension .zip qu'Expand-Archive exige.
$Out = Join-Path $Dir "x"
Remove-Item -Recurse -Force $Out -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $Out | Out-Null
tar -xf $Package -C $Out
if ($LASTEXITCODE -ne 0) { throw "extraction de $Package" }
foreach ($file in "include/WinMLEpCatalog.h", "include/winml/onnxruntime_cxx_api.h",
  "runtimes/win-x64/native/onnxruntime.dll", "runtimes/win-x64/native/Microsoft.Windows.AI.MachineLearning.dll") {
  if (-not (Test-Path (Join-Path $Out $file))) { throw "Windows ML $Version : $file absent du paquet" }
}
if ($env:GITHUB_ENV) { "AJP_WINML_ROOT=$Out" | Out-File -FilePath $env:GITHUB_ENV -Append -Encoding utf8 }
Write-Host "Windows ML $Version : $Out"
