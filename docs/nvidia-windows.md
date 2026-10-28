# TensorRT sous Windows (poste NVIDIA)

Même principe que sous Linux (`docs/nvidia-linux.md`) : le paquet du moteur ne porte que **notre module**
(`ajp_backend_tensorrt.dll`) et `trtexec.exe` ; les bibliothèques de NVIDIA (`nvinfer`, `nvonnxparser`, NVRTC, les ressources
par architecture) se tirent de chez NVIDIA par lien direct (entrées `[[vendor_file]]` de
`tools/release/nvidia-vendor/tensorrt-windows-x64.lock`), après l'acceptation de la licence à l'écran du lecteur, et sont
vérifiées par leur SHA-256.

## Construire et joindre (sur le poste Windows NVIDIA)

```powershell
$env:VCPKG_ROOT = "C:\vcpkg"
$env:AJP_WITH_CUDA = "ON"
$env:AJP_TENSORRT_ROOT = "C:\deps\TensorRT"        # tools/deps/windows-tensorrt.ps1 le prépare (accepte les licences NVIDIA)
$env:AJP_RELEASE_REPO = "Ta2b/ajp-releases"
$env:AJP_RELEASE_TAG = "v0.5.0"
pwsh tools/release/package-windows.ps1 C:\ajp-package
pwsh tools/release/attach-nvidia-windows.ps1 C:\ajp-package v0.5.0            # contrôle
pwsh tools/release/attach-nvidia-windows.ps1 C:\ajp-package v0.5.0 -Publish   # joint à la release
git tag v0.5.0; git push origin v0.5.0                                        # la CI Windows fusionne engines-nvidia.lock
```

Le contrôle refuse une archive qui contient un fichier de NVIDIA, un index qui ne nomme pas cette archive (adresse, SHA-256,
taille) ou sans `[[vendor_file]]`. La CI (`release-windows.yml`) télécharge `engines-nvidia.lock` de la release de
l'étiquette, s'il existe, et en reprend les moteurs dans l'`engines.lock` qu'elle joint.

## À essayer sur le poste Windows (non fait : aucune machine Windows côté développement)

1. `package-windows.ps1` sort `ajp-engine-tensorrt-windows-x64.tar.gz` avec `ajp_backend_tensorrt.dll` et `trtexec.exe`.
2. Installer le moteur depuis l'écran des moteurs : licence, téléchargement des fichiers NVIDIA par plage HTTP, SHA-256 vérifiés.
3. Lire une vidéo ×3 avec SPAN + RIFE : le moteur se construit (trtexec.exe) la première fois, puis la lecture tient.
4. RTX HDR (Maxine) : automatique sous Windows ; vérifier que le SDK s'importe.
