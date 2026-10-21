# Paquets Windows d'une version : un seul lecteur pour tous les GPU, et ses moteurs à la demande
# (même forme que package-linux.sh).
#
#   ajp-windows-x64.zip                     le lecteur, le backend Vulkan, leurs DLL, engines.lock
#                                           et les modèles (models/, sauf AJP_WITHOUT_MODELS)
#   ajp-engine-tensorrt-windows-x64.tar.gz  backend TensorRT, TensorRT et trtexec (NVIDIA), si
#                                           construit (AJP_WITH_CUDA=ON, TensorRT installé)
#   ajp-engine-openvino-windows-x64.tar.gz  backend OpenVINO et OpenVINO (GPU Intel, NPU des Core
#                                           Ultra), si construit (AJP_WITH_OPENVINO=ON, OpenVINO_DIR
#                                           de la roue PyPI)
#
#   $env:VCPKG_ROOT = "C:\vcpkg"; [$env:AJP_WITH_CUDA = "ON"]; [$env:AJP_RELEASE_TAG = "v0.2.0"]
#   pwsh tools/release/package-windows.ps1 [dossier-de-sortie]
#
# À lancer depuis une invite « x64 Native Tools » de Visual Studio (cl.exe dans le PATH). Les DLL
# de vcpkg sont copiées à côté des binaires par la construction (VCPKG_APPLOCAL_DEPS).
param([string]$Out = "")
$ErrorActionPreference = "Stop"

$Root = (Resolve-Path "$PSScriptRoot/../..").Path
if ($Out -eq "") { $Out = Join-Path $Root "build/package" }
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$Out = (Resolve-Path $Out).Path
$Version = (Select-String -Path "$Root/CMakeLists.txt" -Pattern '^  VERSION ([0-9.]+)$').Matches[0].Groups[1].Value
$Tag = if ($env:AJP_RELEASE_TAG) { $env:AJP_RELEASE_TAG } else { "v$Version" }
# Le dépôt dont la release porte les paquets : il doit être public pour que le lecteur les télécharge.
$Repo = if ($env:AJP_RELEASE_REPO) { $env:AJP_RELEASE_REPO } else { "Ta2b/Ultimate-Anime-VideoPlayer" }
$BaseUrl = "https://github.com/$Repo/releases/download/$Tag"
$Cuda = if ($env:AJP_WITH_CUDA) { $env:AJP_WITH_CUDA } else { "OFF" }
$OpenVino = if ($env:AJP_WITH_OPENVINO) { $env:AJP_WITH_OPENVINO } else { "OFF" }

$Extra = @()
# Arbre de construction de vcpkg ailleurs (un chemin court : les sources de certains ports
# dépassent les 260 caractères de MSVC sous build/).
if ($env:AJP_VCPKG_BUILDTREES) { $Extra += "-DVCPKG_INSTALL_OPTIONS=--x-buildtrees-root=$env:AJP_VCPKG_BUILDTREES" }
# Triplet sans Debug (x64-windows-release) : la moitié des ports à compiler sur la CI.
if ($env:AJP_VCPKG_TRIPLET) {
  $Extra += "-DVCPKG_TARGET_TRIPLET=$env:AJP_VCPKG_TRIPLET", "-DVCPKG_HOST_TRIPLET=$env:AJP_VCPKG_TRIPLET"
}
cmake --preset windows-release -S $Root "-DAJP_WITH_CUDA=$Cuda" "-DAJP_WITH_OPENVINO=$OpenVino" @Extra
if ($LASTEXITCODE -ne 0) { throw "configuration" }
cmake --build --preset windows-release -- -k 0  # toutes les erreurs de MSVC en une passe
if ($LASTEXITCODE -ne 0) { throw "construction" }
$Build = Join-Path $Root "build/windows-release"

# --- Le lecteur. ---
$Stage = Join-Path $Out "ajp"
Remove-Item -Recurse -Force $Stage -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $Stage | Out-Null
Copy-Item "$Build/src/app/ajp.exe" $Stage
Copy-Item "$Build/src/app/*.dll" $Stage
Copy-Item "$Build/src/gpu/vulkan/backend/ajp_backend_vulkan.dll" $Stage
Copy-Item "$Build/src/gpu/vulkan/backend/*.dll" $Stage -Exclude "ajp_backend_*" -ErrorAction SilentlyContinue
# Le chargeur Vulkan vient du pilote (vulkan-1.dll dans System32) : jamais livré.
Remove-Item "$Stage/vulkan-1.dll" -ErrorAction SilentlyContinue
if (Test-Path "$Root/LICENSE") { Copy-Item "$Root/LICENSE" $Stage }

# --- Les moteurs. ---
$Lock = Join-Path $Stage "engines.lock"
Set-Content -Path $Lock -Value "version = 1" -Encoding utf8NoBOM

function Add-Engine([string]$Id, [string]$EngineVersion, [string]$License, [string]$Dir) {
  $archive = "ajp-engine-$Id-windows-x64.tar.gz"
  tar -C $Dir -czf (Join-Path $Out $archive) .
  if ($LASTEXITCODE -ne 0) { throw "tar $archive" }
  $file = Get-Item (Join-Path $Out $archive)
  $sha = (Get-FileHash $file -Algorithm SHA256).Hash.ToLower()
  $lines = @("", "[[engine]]", "id = `"$Id`"", "platform = `"windows`"", "url = `"$BaseUrl/$archive`"",
    "sha256 = `"$sha`"", "size = $($file.Length)", "version = `"$EngineVersion`"")
  if ($License -ne "") { $lines += "license = `"$License`"" }
  Add-Content -Path $Lock -Value $lines -Encoding utf8NoBOM
  Write-Host "moteur $Id : $file"
}

$TrtModule = "$Build/src/gpu/cuda/backend/ajp_backend_tensorrt.dll"
if (Test-Path $TrtModule) {
  $dir = Join-Path $Out "engine-tensorrt"
  Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
  New-Item -ItemType Directory -Force -Path $dir | Out-Null
  Copy-Item $TrtModule $dir
  # TensorRT et ses ressources du constructeur, depuis le dossier lib de AJP_TENSORRT_ROOT.
  $cache = Get-Content "$Build/CMakeCache.txt"
  $trtRoot = ($cache | Select-String '^AJP_TENSORRT_ROOT:[A-Z]*=(.*)$').Matches[0].Groups[1].Value
  foreach ($pattern in "nvinfer*.dll", "nvonnxparser*.dll") {
    Copy-Item "$trtRoot/lib/$pattern" $dir -ErrorAction SilentlyContinue
    Copy-Item "$trtRoot/bin/$pattern" $dir -ErrorAction SilentlyContinue
  }
  Copy-Item "$trtRoot/bin/trtexec.exe" $dir -ErrorAction SilentlyContinue
  # Le runtime CUDA lié au backend (cudart64_*.dll), du kit CUDA.
  if ($env:CUDA_PATH) { Copy-Item "$env:CUDA_PATH/bin/cudart64_*.dll" $dir -ErrorAction SilentlyContinue }
  $nvinfer = Get-ChildItem "$dir/nvinfer_*.dll" | Where-Object { $_.Name -match '^nvinfer_(\d+)\.dll$' } | Select-Object -First 1
  $trtVersion = if ($nvinfer) { $Matches[1] } else { $Version }
  Add-Engine "tensorrt" $trtVersion "https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/sla.html" $dir
}

$OvModule = "$Build/src/gpu/openvino/ajp_backend_openvino.dll"
if (Test-Path $OvModule) {
  $dir = Join-Path $Out "engine-openvino"
  Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
  New-Item -ItemType Directory -Force -Path $dir | Out-Null
  Copy-Item $OvModule $dir
  # OpenVINO depuis les bibliothèques de sa roue (OpenVINO_DIR = <roue>/openvino/cmake) : le
  # runtime et TBB, les greffons GPU et NPU (avec le compilateur du NPU), le lecteur ONNX et
  # cache.json (noyaux réglés du greffon GPU), ouverts depuis le dossier de openvino.dll.
  $ovLibs = Join-Path (Split-Path $env:OpenVINO_DIR -Parent) "libs"
  foreach ($pattern in "openvino.dll", "openvino_intel_gpu_plugin.dll", "openvino_intel_npu_*.dll",
    "openvino_onnx_frontend.dll", "tbb12.dll", "tbbbind_2_5.dll", "tbbmalloc.dll", "cache.json") {
    Copy-Item "$ovLibs/$pattern" $dir
  }
  Add-Engine "openvino" $Version "" $dir
}

# --- Les modèles SPAN et RIFE livrés (comme sous Linux) : le bash de Git, pas celui de WSL. ---
if (-not $env:AJP_WITHOUT_MODELS) {
  $bash = Join-Path $env:ProgramFiles "Git/bin/bash.exe"
  if (-not (Test-Path $bash)) { $bash = "bash" }
  $script = "$Root/tools/release/fetch-models.sh".Replace('\', '/')
  $models = (Join-Path $Stage "models").Replace('\', '/')
  & $bash $script $models
  if ($LASTEXITCODE -ne 0) { throw "modèles" }
}

Compress-Archive -Path $Stage -DestinationPath (Join-Path $Out "ajp-windows-x64.zip") -Force
Get-ChildItem $Out -File | Where-Object { $_.Name -like "ajp-*" -and $_.Extension -ne ".txt" } | ForEach-Object {
  "$((Get-FileHash $_ -Algorithm SHA256).Hash.ToLower())  $($_.Name)"
} | Set-Content -Path (Join-Path $Out "SHA256SUMS-windows.txt") -Encoding utf8NoBOM
Write-Host "Paquets : $Out"
