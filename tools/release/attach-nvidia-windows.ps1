# Joint le paquet TensorRT construit sur un poste Windows NVIDIA à la release d'une étiquette, AVANT de pousser
# l'étiquette (comme attach-nvidia-linux.sh). release-windows.yml télécharge ensuite engines-nvidia.lock de la release
# et en fusionne les moteurs (et leurs [[vendor_file]]) dans l'engines.lock qu'il joint (package-windows.ps1,
# AJP_ENGINES_LOCK) ; l'archive du moteur reste celle du poste, son SHA-256 est déjà dans l'index.
#
#   1. sur le poste, à l'étiquette visée (le SHA-256 dépend de l'archive : pas de commit entre-temps) :
#        $env:AJP_WITH_CUDA = "ON"; $env:AJP_TENSORRT_ROOT = "<racine TensorRT>"
#        $env:AJP_RELEASE_REPO = "<dépôt des releases>"; $env:AJP_RELEASE_TAG = "v0.5.0"
#        pwsh tools/release/package-windows.ps1 <dossier>
#   2. pwsh tools/release/attach-nvidia-windows.ps1 <dossier> v0.5.0              # contrôle seulement
#      pwsh tools/release/attach-nvidia-windows.ps1 <dossier> v0.5.0 -Publish    # crée la release si besoin et joint
#   3. git tag v0.5.0; git push origin v0.5.0                                      # la CI fait le reste
param(
  [Parameter(Mandatory = $true)][string]$Dir,
  [Parameter(Mandatory = $true)][string]$Tag,
  [switch]$Publish
)
$ErrorActionPreference = "Stop"

$archive = Join-Path $Dir "ajp-engine-tensorrt-windows-x64.tar.gz"
$lock = Join-Path $Dir "engines.lock"
if (-not (Test-Path $archive)) { throw "$archive absent : le paquet TensorRT n'a pas été construit (AJP_WITH_CUDA=ON)" }
if (-not (Test-Path $lock)) { throw "$lock absent" }
$repo = if ($env:AJP_RELEASE_REPO) { $env:AJP_RELEASE_REPO } else { (gh repo view --json nameWithOwner -q .nameWithOwner) }

# Rien de NVIDIA dans l'archive : seulement notre module et trtexec.exe (le lien direct tire le reste).
$members = tar tzf $archive
$nvidia = $members | Where-Object { $_ -match '(?i)nvinfer|nvonnxparser|nvrtc|cudart|cublas|cudnn|builder_resource' }
if ($nvidia) { throw ("L'archive contient des fichiers NVIDIA (construite avec AJP_NVIDIA_BUNDLE ?) :`n" + ($nvidia -join "`n")) }

$sha = (Get-FileHash $archive -Algorithm SHA256).Hash.ToLower()
$size = (Get-Item $archive).Length
$url = "https://github.com/$repo/releases/download/$Tag/ajp-engine-tensorrt-windows-x64.tar.gz"
$text = Get-Content $lock
foreach ($want in "url = `"$url`"", "sha256 = `"$sha`"", "size = $size") {
  if (-not ($text -contains $want)) { throw "$lock ne contient pas : $want (étiquette ou dépôt différents de ceux de la construction ?)" }
}
$vendor = ($text | Where-Object { $_ -eq "[[vendor_file]]" }).Count
if ($vendor -lt 1) { throw "$lock n'a aucun [[vendor_file]] : sans eux le moteur s'installerait sans les bibliothèques de NVIDIA" }

# engines-nvidia.lock : les moteurs et fichiers de NVIDIA seulement (TensorRT, et RTX HDR s'il est là).
$out = Join-Path $Dir "engines-nvidia.lock"
$blocks = (Get-Content $lock -Raw) -split '(?m)^(?=\[\[(?:engine|vendor_file)\]\]\r?$)' | Where-Object { $_ -match '^\[\[(engine|vendor_file)\]\]' }
$kept = @("version = 1")
foreach ($block in $blocks) {
  $id = [regex]::Match($block, '(?m)^id = "(.*)"\r?$').Groups[1].Value
  if ($id -eq "tensorrt" -or $id -eq "maxine_vfx") { $kept += @("", $block.TrimEnd()) }
}
Set-Content -Path $out -Value $kept -Encoding utf8NoBOM
Write-Host "archive      : $archive ($size octets, sha256 $sha)"
Write-Host "index NVIDIA : $out"
Write-Host "release      : $repo $Tag"
if (-not $Publish) { Write-Host "Contrôle seulement. Pour joindre : relancer avec -Publish, puis pousser l'étiquette $Tag."; exit 0 }
gh release view $Tag -R $repo *> $null
if ($LASTEXITCODE -ne 0) { gh release create $Tag -R $repo --prerelease --title $Tag --notes $Tag }
gh release upload $Tag -R $repo --clobber $archive $out
Write-Host "Joint à $repo $Tag. Pousser maintenant l'étiquette : git tag $Tag; git push origin $Tag"
