# Paquets Windows d'une version : un seul lecteur pour tous les GPU, et ses moteurs à la demande
# (même forme que package-linux.sh).
#
#   kirari-windows-x64.zip                     le lecteur, le backend Vulkan, leurs DLL, engines.lock
#                                           et les modèles (models/, sauf KIRARI_WITHOUT_MODELS)
#   kirari-engine-tensorrt-windows-x64.tar.gz  backend TensorRT et trtexec (NVIDIA), si
#                                           construit (KIRARI_WITH_CUDA=ON, TensorRT installé)
#
#   $env:VCPKG_ROOT = "C:\vcpkg"; [$env:KIRARI_WITH_CUDA = "ON"]; [$env:KIRARI_RELEASE_TAG = "v0.2.0"]
#   pwsh tools/release/package-windows.ps1 [dossier-de-sortie] [-TensorRtOnly]
#
# À lancer depuis une invite « x64 Native Tools » de Visual Studio (cl.exe dans le PATH). Les DLL
# de vcpkg sont copiées à côté des binaires par la construction (VCPKG_APPLOCAL_DEPS).
param([string]$Out = "", [switch]$TensorRtOnly)
$ErrorActionPreference = "Stop"

$Root = (Resolve-Path "$PSScriptRoot/../..").Path
if ($Out -eq "") { $Out = Join-Path $Root "build/package" }
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$Out = (Resolve-Path $Out).Path
$Version = (Select-String -Path "$Root/CMakeLists.txt" -Pattern '^  VERSION ([0-9.]+)$').Matches[0].Groups[1].Value
$Tag = if ($env:KIRARI_RELEASE_TAG) { $env:KIRARI_RELEASE_TAG } else { "v$Version" }
# Le dépôt dont la release porte les paquets : il doit être public pour que le lecteur les télécharge.
# Une seule source pour le dépôt par défaut : tools/release/versions.env (A28-05).
$Repo = if ($env:KIRARI_RELEASE_REPO) { $env:KIRARI_RELEASE_REPO } else {
  (Select-String -Path "$PSScriptRoot/versions.env" -Pattern '^KIRARI_RELEASE_REPO=(.+)$').Matches[0].Groups[1].Value
}
$BaseUrl = "https://github.com/$Repo/releases/download/$Tag"
$Cuda = if ($env:KIRARI_WITH_CUDA) { $env:KIRARI_WITH_CUDA } else { "OFF" }
if ($TensorRtOnly -and $Cuda -ne "ON") { throw "-TensorRtOnly demande KIRARI_WITH_CUDA=ON" }

$Extra = @()
# Arbre de construction de vcpkg ailleurs (un chemin court : les sources de certains ports
# dépassent les 260 caractères de MSVC sous build/).
if ($env:KIRARI_VCPKG_BUILDTREES) { $Extra += "-DVCPKG_INSTALL_OPTIONS=--x-buildtrees-root=$env:KIRARI_VCPKG_BUILDTREES" }
# Un build TensorRT réutilisé peut avoir servi à un essai CUTLASS. Rétablir le défaut Windows à chaque appel.
if ($Cuda -eq "ON") {
  $SpanFp8 = if ($env:KIRARI_WITH_SPAN_FP8) { $env:KIRARI_WITH_SPAN_FP8 } else { "OFF" }
  $Extra += "-DKIRARI_WITH_SPAN_FP8:BOOL=$SpanFp8"
}
if ($Cuda -eq "ON" -and $env:KIRARI_TENSORRT_ROOT) { $Extra += "-DKIRARI_TENSORRT_ROOT=$env:KIRARI_TENSORRT_ROOT" }
if ($Cuda -eq "ON" -and $env:KIRARI_TRTEXEC) { $Extra += "-DKIRARI_TRTEXEC:FILEPATH=$env:KIRARI_TRTEXEC" }
# sccache sur la CI (release-windows.yml) : les objets de cl.exe repris d'une construction à l'autre.
if ($env:KIRARI_COMPILER_LAUNCHER) {
  $Extra += "-DCMAKE_C_COMPILER_LAUNCHER=$env:KIRARI_COMPILER_LAUNCHER", "-DCMAKE_CXX_COMPILER_LAUNCHER=$env:KIRARI_COMPILER_LAUNCHER"
}
# Un paquet complet réactive les tests après une release rapide dans le même build.
$BuildTests = if ($TensorRtOnly) { "OFF" } elseif ($env:KIRARI_BUILD_TESTS) { $env:KIRARI_BUILD_TESTS } else { "ON" }
$Extra += "-DKIRARI_BUILD_TESTS:BOOL=$BuildTests"
# Triplet sans Debug (x64-windows-release) : la moitié des ports à compiler sur la CI.
if ($env:KIRARI_VCPKG_TRIPLET) {
  $Extra += "-DVCPKG_TARGET_TRIPLET=$env:KIRARI_VCPKG_TRIPLET", "-DVCPKG_HOST_TRIPLET=$env:KIRARI_VCPKG_TRIPLET"
}
# Une étiquette de préversion (v0.11.0-rc.1) donne son suffixe au binaire : il dit la version de l'étiquette (A26-04).
$Suffix = if ($Tag.StartsWith("v$Version-")) { $Tag.Substring("v$Version".Length) } else { "" }
$BuildTargets = @()
$Build = Join-Path $Root "build/windows-release"
if ($TensorRtOnly) {
  $Build = Join-Path $Root "build/windows-release-tensorrt"
  $BuildTargets = @("--target", "kirari_backend_tensorrt")
}
cmake --preset windows-release -S $Root -B $Build "-DKIRARI_WITH_CUDA=$Cuda" "-DKIRARI_VERSION_SUFFIX=$Suffix" @Extra
if ($LASTEXITCODE -ne 0) { throw "configuration" }
cmake --build $Build @BuildTargets -- -k 0  # toutes les erreurs de MSVC en une passe
if ($LASTEXITCODE -ne 0) { throw "construction" }

# --- Le lecteur. ---
$Stage = Join-Path $Out "kirari"
if (-not $TensorRtOnly) {
  Remove-Item -Recurse -Force $Stage -ErrorAction SilentlyContinue
  New-Item -ItemType Directory -Force -Path $Stage | Out-Null
  Copy-Item "$Build/src/app/kirari.exe" $Stage
  Copy-Item "$Build/src/app/*.dll" $Stage
  Copy-Item "$Root/LICENSE", "$Root/THIRD_PARTY_NOTICES.md" $Stage
  Copy-Item (Join-Path $PSScriptRoot "windows/LISEZMOI.txt") $Stage
  Copy-Item "$Build/src/gpu/vulkan/backend/kirari_backend_vulkan.dll" $Stage
  Copy-Item "$Build/src/gpu/vulkan/backend/*.dll" $Stage -Exclude "kirari_backend_*" -ErrorAction SilentlyContinue
  # Le runtime C++ de Visual Studio (CRT dynamique, triplet x64-windows-release) est déployé à côté de
  # kirari.exe. Ces DLL servent aussi aux modules chargés dans le même processus (C51-05).
  if (-not $env:VCToolsRedistDir) { throw "VCToolsRedistDir absent : lancer depuis une invite x64 Native Tools (runtime C++ à livrer)" }
  $crt = Get-ChildItem (Join-Path $env:VCToolsRedistDir "x64") -Directory -Filter "Microsoft.VC*.CRT" | Select-Object -First 1
  if (-not $crt) { throw "runtime C++ introuvable dans $env:VCToolsRedistDir\x64" }
  Copy-Item "$($crt.FullName)/*.dll" $Stage
  foreach ($name in "vcruntime140.dll", "msvcp140.dll") {
    if (-not (Test-Path (Join-Path $Stage $name))) { throw "$name absent du runtime C++ copié" }
  }
  # Le chargeur Vulkan vient du pilote (vulkan-1.dll dans System32) : jamais livré.
  Remove-Item "$Stage/vulkan-1.dll" -ErrorAction SilentlyContinue
  Copy-Item "$Root/data/luts" (Join-Path $Stage "luts") -Recurse
}

# --- Les moteurs. ---
$LockDirectory = if ($TensorRtOnly) { $Out } else { $Stage }
$Lock = Join-Path $LockDirectory "engines.lock"
Set-Content -Path $Lock -Value "version = 1" -Encoding utf8NoBOM

function Add-Engine([string]$Id, [string]$EngineVersion, [string]$License, [string]$Dir) {
  $archive = "kirari-engine-$Id-windows-x64.tar.gz"
  # Notre module est signé comme le reste (B51-05) ; trtexec.exe est celui de NVIDIA, laissé tel quel : le module en
  # garde l'empreinte. Sans compte de signature, sign.ps1 ne fait rien.
  $module = Join-Path $Dir "kirari_backend_$Id.dll"
  if (Test-Path $module) {
    pwsh (Join-Path $PSScriptRoot "windows/sign.ps1") $module
    if ($LASTEXITCODE -ne 0) { throw "signature de $module" }
  }
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

$TrtModule = "$Build/src/gpu/cuda/backend/kirari_backend_tensorrt.dll"
if ($Cuda -eq "ON") {
  if (-not (Test-Path $TrtModule)) { throw "module TensorRT absent ($TrtModule)" }
  $dir = Join-Path $Out "engine-tensorrt"
  Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
  New-Item -ItemType Directory -Force -Path $dir | Out-Null
  Copy-Item $TrtModule $dir
  $cache = Get-Content "$Build/CMakeCache.txt" -Raw
  $trtRoot = [regex]::Match($cache, '(?m)^KIRARI_TENSORRT_ROOT:PATH=([^\r\n]*)').Groups[1].Value
  # Le fichier archivé doit être celui dont CMake a gravé le SHA-256 dans le module.
  $trtexec = [regex]::Match($cache, '(?m)^KIRARI_TRTEXEC_PACKAGED:INTERNAL=([^\r\n]*)').Groups[1].Value
  if (-not $trtexec -or -not (Test-Path -LiteralPath $trtexec -PathType Leaf)) {
    throw "trtexec.exe absent de la configuration du module TensorRT ($trtexec)"
  }
  Copy-Item -LiteralPath $trtexec -Destination (Join-Path $dir "trtexec.exe")
  # La version entière de TensorRT (11.3.0.99), celle de versions.env : jamais le seul numéro majeur, ni la version
  # du lecteur par défaut (A28-09). Elle nomme le dossier d'installation.
  $trtVersion = (Select-String -Path "$PSScriptRoot/versions.env" -Pattern '^TENSORRT_LIBS_VERSION=(.+)$').Matches[0].Groups[1].Value
  if (-not $trtVersion) { throw "TENSORRT_LIBS_VERSION absent de versions.env" }
  $vendor = "$Root/tools/release/nvidia-vendor/tensorrt-windows-x64.lock"
  if (-not $env:KIRARI_NVIDIA_BUNDLE -and (Test-Path $vendor)) {
    # Lien direct : le paquet ne porte que le module ; les bibliothèques de NVIDIA et trtexec se
    # tirent des archives de NVIDIA elles-mêmes, après l'acceptation de sa licence à l'écran
    # (entrées [[vendor_file]], tools/release/vendor-index.py).
    Add-Engine "tensorrt" $trtVersion "https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/sla.html" $dir
    Add-Content -Path $Lock -Value (@("") + (Get-Content $vendor)) -Encoding utf8NoBOM
  } else {
    # TensorRT et ses ressources du constructeur, depuis le dossier lib de KIRARI_TENSORRT_ROOT.
    foreach ($pattern in "nvinfer*.dll", "nvonnxparser*.dll") {
      Copy-Item "$trtRoot/lib/$pattern" $dir -ErrorAction SilentlyContinue
      Copy-Item "$trtRoot/bin/$pattern" $dir -ErrorAction SilentlyContinue
    }
    # Le runtime CUDA lié au backend (cudart64_*.dll), du kit CUDA.
    if ($env:CUDA_PATH) {
      Copy-Item "$env:CUDA_PATH/bin/cudart64_*.dll" $dir -ErrorAction SilentlyContinue
      Copy-Item "$env:CUDA_PATH/bin/x64/cudart64_*.dll" $dir -ErrorAction SilentlyContinue
    }
    Add-Engine "tensorrt" $trtVersion "https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/sla.html" $dir
  }
}

if ($TensorRtOnly) {
  Write-Host "Paquet TensorRT : $Out"
  return
}

# --- Les modèles SPAN et RIFE livrés (comme sous Linux) : le bash de Git, pas celui de WSL. ---
if (-not $env:KIRARI_WITHOUT_MODELS) {
  $bash = Join-Path $env:ProgramFiles "Git/bin/bash.exe"
  if (-not (Test-Path $bash)) { $bash = "bash" }
  $script = "$Root/tools/release/fetch-models.sh".Replace('\', '/')
  $models = (Join-Path $Stage "models").Replace('\', '/')
  & $bash $script $models
  if ($LASTEXITCODE -ne 0) { throw "modèles" }
}

# KIRARI_ENGINES_LOCK=<fichier ou lien de l'engines.lock d'un poste NVIDIA> reprend ses moteurs (et leurs fichiers
# [[vendor_file]]) dont l'id manque ici, comme package-linux.sh : le poste Windows NVIDIA joint les siens à la release
# avant l'étiquette (tools/release/attach-nvidia-windows.ps1), la CI les fusionne dans l'index qu'elle joint.
if ($env:KIRARI_ENGINES_LOCK) {
  $other = $env:KIRARI_ENGINES_LOCK
  if ($other -match '^https?://') {
    $other = Join-Path $Out "engines-autre.lock"
    Invoke-WebRequest -Uri $env:KIRARI_ENGINES_LOCK -OutFile $other -UseBasicParsing
  }
  # Fins de ligne CRLF sous Windows : ramenées à LF, sans quoi « $ » ne tombe jamais en fin de ligne.
  $have = [regex]::Matches(((Get-Content $Lock -Raw) -replace "`r`n", "`n"), '(?m)^id = "(.*)"$') | ForEach-Object { $_.Groups[1].Value }
  $blocks = ((Get-Content $other -Raw) -replace "`r`n", "`n") -split '(?m)^(?=\[\[(?:engine|vendor_file)\]\]$)' | Where-Object { $_ -match '^\[\[(engine|vendor_file)\]\]' }
  foreach ($block in $blocks) {
    $id = [regex]::Match($block, '(?m)^id = "(.*)"$').Groups[1].Value
    if ($id -and -not ($have -contains $id)) { Add-Content -Path $Lock -Value ("", $block.TrimEnd()) -Encoding utf8NoBOM }
  }
  Write-Host ("moteurs d'engines.lock : " + (([regex]::Matches(((Get-Content $Lock -Raw) -replace "`r`n", "`n"), '(?m)^id = "(.*)"$') | ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique) -join " "))
}
Copy-Item $Lock (Join-Path $Out "engines.lock") -Force

# Signature Authenticode du lecteur et de ses DLL avant l'archive (P7.1 ; sans effet sans certificat).
pwsh (Join-Path $PSScriptRoot "windows/sign.ps1") $Stage
if ($LASTEXITCODE -ne 0) { throw "signature" }
Compress-Archive -Path $Stage -DestinationPath (Join-Path $Out "kirari-windows-x64.zip") -Force
Get-ChildItem $Out -File | Where-Object { $_.Name -like "kirari-*" -and $_.Extension -ne ".txt" } | ForEach-Object {
  "$((Get-FileHash $_ -Algorithm SHA256).Hash.ToLower())  $($_.Name)"
} | Set-Content -Path (Join-Path $Out "SHA256SUMS-windows.txt") -Encoding utf8NoBOM
Write-Host "Paquets : $Out"
