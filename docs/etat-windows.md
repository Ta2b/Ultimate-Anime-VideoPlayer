# État Windows NVIDIA (poste RTX 4070 Laptop 8 Go) — 2026-10-31

Reprise du travail : ce qui est fait, mesuré, et ce qui reste. Rien n'est publié (ni release, ni `ajp-releases`, ni étiquette).

## Fait
- main se construit sous Windows (preset `windows-release`, vcpkg + MSVC, CUDA 13.4 / TensorRT 11.3 activés) sans changement de code.
  Pièges rencontrés : `clang-cl` requis par le port libplacebo (composant « Clang » de VS Build Tools), `C:\vcpkg` doit
  être un clone complet (pas `--depth 1`), `VCPKG_ROOT` à poser **après** le shell VS, `CC=cl CXX=cl` avec LLVM en fin de PATH,
  `7z` dans le PATH pour l'étape des modèles de `package-windows.ps1`.
- Paquet local (non publié) : `ajp-windows-x64.zip` et `ajp-engine-tensorrt-windows-x64.tar.gz` (module + `trtexec.exe`, aucun fichier NVIDIA).
- `ajp_backend_tensorrt.dll` se charge ; ses 21 tests CUDA passent sur la 4070 (il faut `spdlog.dll` et les DLL TensorRT dans le PATH des tests).
- Télémétrie GPU par NVML sous Windows (`nvml.dll` du pilote) : écrite, compilée, symboles vérifiés ; à voir en lecture (onglet Stats).
- DLL factices des tests `vendor_hdr` exportées sous MSVC : 18 échecs de tests → 11 (restent des tests Maxine/Linux qui attendent des `.so`).

## Constats
- Écran d'accueil du premier lancement : assistant en 7 étapes qui attend des clics ; sauté si `first_run_done = true` dans `settings.toml`.
  Idée : ne pas bloquer quand une vidéo est passée en argument.
- Lecture Vulkan seule (Re:Zero / Slime 1080p, RIFE ×3 + SPAN ×2 → 2560×1440, sortie SDR, VRR 240 Hz) : tient environ 24 fps sur les scènes lourdes ; le mode Auto redescend.
- Journal : « l'appareil n'exporte pas ses images ni ses sémaphores (fd) : ring sans poignées » — la synchronisation partagée Vulkan↔CUDA est désactivée sous Windows
  (`shared_tensor.cpp`, `exports_semaphores` toujours faux). Côté CUDA le chemin Win32 existe déjà (`ring_slot.cpp`).
- Premier moteur SPAN : écran noir avec « Preparing engine SPAN » plusieurs secondes.
- LUT par défaut (`HDR grading base…cube`, etc.) introuvables : fichiers de l'utilisateur à déposer dans le dossier `luts` des données.

## À faire
1. Essai de lecture **TensorRT** (SPAN + RIFE ×3) : le premier essai est resté en Vulkan (« TensorRT is installed but the player uses vulkan »).
   Il faut choisir TensorRT dans Réglages > Moteurs (ou `settings.toml` : `[engines] chosen/installed` et `[engines.preferred]`, mais le lecteur réécrit le fichier).
   Installation locale de test : `<données>\backends\tensorrt\11\` avec le module, `trtexec.exe` et les DLL NVIDIA copiées de `C:\deps` (jamais dans un paquet).
2. Sémaphores timeline partagés Vulkan↔CUDA par handles Win32 (`VK_KHR_external_semaphore_win32` + `TimelineSemaphoreWin32` côté CUDA), validés par un test.
3. Mesures de performance Windows : présentation (flip direct, swapchain attendable), priorités de files GPU.
4. RTX HDR Windows : reporté ; HDR adaptatif pour tous. Le pont `aji_rtx.dll` (SDK RTX Video, licence NVIDIA à accepter) n'existe pas dans les dépôts de Ta2b (seul le pont Linux/Maxine y est).
5. Cache vcpkg sur GitHub Actions (`actions/cache`) et triplet `x64-windows-release` pour raccourcir la CI Windows.
6. Essai Re:Zero ép. 18 (`Téléchargements`), grade adaptatif, LUT, HDR auto fast.
