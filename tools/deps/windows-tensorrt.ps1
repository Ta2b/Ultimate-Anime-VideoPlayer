# Le kit CUDA (nvcc, runtime, en-têtes) et TensorRT pour construire ajp_backend_tensorrt sous
# Windows, aux versions de tools/release/versions.env, dans le dossier donné. Le lancer, c'est
# accepter la licence du kit CUDA (https://docs.nvidia.com/cuda/eula/index.html) et celle de
# TensorRT (https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/sla.html) : la CI ne le
# fait que sur l'entrée tensorrt de release-windows.yml, que le propriétaire du dépôt coche.
#
# Le kit vient des archives redist de NVIDIA, chacune vérifiée par l'empreinte de son manifeste,
# fusionnées en un seul dossier (bin/, include/, lib/x64/, nvvm/) ; TensorRT, de son zip.
# Sous GitHub Actions, CUDA_PATH, CUDACXX, AJP_TENSORRT_ROOT et le PATH passent aux étapes
# suivantes.
#
#   pwsh tools/deps/windows-tensorrt.ps1 C:\deps\nvidia
param([Parameter(Mandatory = $true)][string]$Dir)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"  # Invoke-WebRequest est dix fois plus lent avec la barre

$Root = (Resolve-Path "$PSScriptRoot/../..").Path
$Pins = Get-Content "$Root/tools/release/versions.env"
function Pinned([string]$Name) {
  $line = $Pins | Select-String -Pattern "^$Name=(\S+)$"
  if (-not $line) { throw "$Name absent de tools/release/versions.env" }
  return $line.Matches[0].Groups[1].Value
}
$Series = Pinned "CUDA_REDIST_SERIES"
$TrtUrl = Pinned "TENSORRT_WINDOWS_URL"

New-Item -ItemType Directory -Force -Path $Dir | Out-Null
$Dir = (Resolve-Path $Dir).Path
$Downloads = Join-Path $Dir "downloads"
New-Item -ItemType Directory -Force -Path $Downloads | Out-Null

# --- Le kit CUDA : le dernier manifeste redistrib_<série>.<n>.json. ---
$Redist = "https://developer.download.nvidia.com/compute/cuda/redist"
$index = (Invoke-WebRequest -Uri "$Redist/" -UseBasicParsing).Content
$releases = [regex]::Matches($index, "redistrib_($([regex]::Escape($Series))\.\d+)\.json") |
  ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique { [version]$_ }
if (-not $releases) { throw "aucun manifeste CUDA $Series dans $Redist" }
$Cuda = @($releases)[-1]
$Manifest = Invoke-RestMethod -Uri "$Redist/redistrib_$Cuda.json"
Write-Host "CUDA $Cuda (manifeste redistrib_$Cuda.json)"

$CudaRoot = Join-Path $Dir "cuda"
Remove-Item -Recurse -Force $CudaRoot -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $CudaRoot | Out-Null
# nvcc et son frontal (nvvm, crt), le runtime et le pilote (cudart.lib, cuda.lib), les en-têtes
# de CCCL ; les composants absents d'une version (crt, nvvm séparés depuis CUDA 13) sont passés.
# Un composant renommé garde ses deux noms, l'ancien d'abord : cuda_cccl est devenu cccl.
foreach ($names in @("cuda_nvcc"), @("cuda_crt"), @("libnvvm"), @("cuda_cudart"), @("cuda_cccl", "cccl"), @("cuda_nvrtc")) {
  $component = $names | Where-Object { $Manifest.$_ -and $Manifest.$_."windows-x86_64" } | Select-Object -First 1
  if (-not $component) {
    Write-Host "  $($names -join ' ou ') : absent de CUDA $Cuda"
    continue
  }
  $entry = $Manifest.$component
  $archive = $entry."windows-x86_64"
  $file = Join-Path $Downloads (Split-Path $archive.relative_path -Leaf)
  if (-not (Test-Path $file)) { Invoke-WebRequest -Uri "$Redist/$($archive.relative_path)" -OutFile $file -UseBasicParsing }
  $sha = (Get-FileHash $file -Algorithm SHA256).Hash.ToLower()
  if ($sha -ne $archive.sha256) { Remove-Item $file; throw "$component : empreinte $sha, attendue $($archive.sha256)" }
  $unpacked = Join-Path $Downloads $component
  Remove-Item -Recurse -Force $unpacked -ErrorAction SilentlyContinue
  New-Item -ItemType Directory -Force -Path $unpacked | Out-Null
  tar -xf $file -C $unpacked
  if ($LASTEXITCODE -ne 0) { throw "extraction de $file" }
  # Chaque archive a un dossier racine <composant>-windows-x86_64-<version>-archive.
  $top = Get-ChildItem $unpacked -Directory | Select-Object -First 1
  Copy-Item -Path (Join-Path $top.FullName "*") -Destination $CudaRoot -Recurse -Force
  Write-Host "  $component $($entry.version) : $((Get-ChildItem $top.FullName -Name) -join ', ')"
}
$Nvcc = Join-Path $CudaRoot "bin/nvcc.exe"
if (-not (Test-Path $Nvcc)) { throw "nvcc.exe absent de $CudaRoot" }
# nvcc cherche CCCL dans include/cccl depuis CUDA 13, et CMake refuse une cible importée
# (CUDA::cudart) dont un dossier d'en-têtes manque : CCCL doit y être, ou à plat dans include/.
$Include = Join-Path $CudaRoot "include"
$Cccl = Join-Path $Include "cccl"
if (Test-Path (Join-Path $Cccl "cuda/std")) {
  Write-Host "  CCCL : include/cccl"
} elseif (Test-Path (Join-Path $Include "cuda/std")) {
  New-Item -ItemType Directory -Force -Path $Cccl | Out-Null
  Write-Host "  CCCL : à plat dans include/"
} else {
  Get-ChildItem $Include -Directory -Name | ForEach-Object { Write-Host "  include/$_" }
  throw "en-têtes de CCCL (cuda/std) absents de $Include"
}

# --- TensorRT : son zip, dont le dossier racine porte la version (TensorRT-11.3.0.99). ---
$TrtZip = Join-Path $Downloads (Split-Path $TrtUrl -Leaf)
if (-not (Test-Path $TrtZip)) { Invoke-WebRequest -Uri $TrtUrl -OutFile $TrtZip -UseBasicParsing }
$TrtSha = (Get-FileHash $TrtZip -Algorithm SHA256).Hash.ToLower()
Write-Host "TensorRT : $(Split-Path $TrtUrl -Leaf), $((Get-Item $TrtZip).Length) octets, sha256 $TrtSha"
$TrtUnpacked = Join-Path $Dir "tensorrt"
Remove-Item -Recurse -Force $TrtUnpacked -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $TrtUnpacked | Out-Null
tar -xf $TrtZip -C $TrtUnpacked
if ($LASTEXITCODE -ne 0) { throw "extraction de $TrtZip" }
$TrtRoot = (Get-ChildItem $TrtUnpacked -Directory | Where-Object { Test-Path (Join-Path $_.FullName "include/NvInfer.h") } |
  Select-Object -First 1).FullName
if (-not $TrtRoot) { throw "NvInfer.h absent du zip de TensorRT" }
Get-ChildItem "$TrtRoot/lib", "$TrtRoot/bin" -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "  $($_.Name)" }

if ($env:GITHUB_ENV) {
  "CUDA_PATH=$CudaRoot", "CUDACXX=$Nvcc", "AJP_TENSORRT_ROOT=$TrtRoot" |
    Out-File -FilePath $env:GITHUB_ENV -Append -Encoding utf8
  (Join-Path $CudaRoot "bin"), (Join-Path $CudaRoot "bin/x64"), "$TrtRoot/lib", "$TrtRoot/bin" |
    Out-File -FilePath $env:GITHUB_PATH -Append -Encoding utf8
}
Write-Host "CUDA : $CudaRoot ; TensorRT : $TrtRoot"
