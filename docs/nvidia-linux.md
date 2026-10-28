# NVIDIA sous Linux, jusqu'à la RTX 4070 Ti

Ce guide suit l'audit « NVIDIA Linux » (`/mnt/project-files/audit/audit-nvidia-linux.html`, constats
LN-01 à LN-20). Il dit ce que le lecteur fait seul sur une carte NVIDIA, ce qui se construit sur la
machine, et ce qu'il reste à mesurer.

## Deux voies

- **Vulkan** : livrée à tous. Décodage Vulkan Video, réseaux en shaders (matrices coopératives,
  préréglages Ada pour SPAN 720p et RIFE 4.25 dans `tools/vk/presets/`).
- **TensorRT + CUDA + NVDEC** : la voie de performance maximale. Nos releases ne redistribuent pas
  les bibliothèques de NVIDIA (licence TensorRT, clause 12.1 : seules libnvinfer et libnvinfer_plugin
  sont redistribuables, avec des conditions). L'appli les **télécharge chez NVIDIA** (les roues
  de pypi.nvidia.com) après l'acceptation de la licence à l'écran des moteurs : notre paquet ne
  porte que le module et le lecteur n'en tire que les fichiers utiles à la carte (libnvinfer, le
  lecteur ONNX, les ressources de l'architecture du GPU, NVRTC), par plages HTTP, chacun vérifié par
  son SHA-256 (`[[vendor_file]]` d'engines.lock ; `tools/release/vendor-index.py` les écrit). Sans
  index (une release sans paquet NVIDIA), la construction sur la machine reste possible.

## Construire la voie TensorRT sur la machine

Prérequis : kit CUDA 12.x, TensorRT 10 ou 11 (paquets NVIDIA), pilote récent.

```
export AJP_DEPS_PREFIX=...                       # préfixe des dépendances (tools/deps/linux-from-source.sh)
cmake --preset linux-prefix-cuda-machine         # CUDA sm_89, -O3, LTO, x86-64-v3 (CPU récent)
cmake --build --preset linux-prefix-cuda-machine
```

`linux-prefix-cuda-release` reste le préréglage sans LTO ni `-march`. Le préréglage « machine » ne
change rien aux noyaux GPU : il accélère les threads source et préparation. Pour une autre carte,
changer `AJP_CUDA_ARCHITECTURES` (`89-real` pour la 4070 Ti).

RTX HDR (Maxine) s'importe depuis le SDK installé : `--truehdr-sdk /usr/local/VideoFX`.

## Ce que le lecteur fait seul sur une carte NVIDIA

| Réglage | Effet | Désactivation |
|---|---|---|
| Voie CUDA de l'agrandissement en haute priorité (LN-04) | le GPU lui donne la main en premier | aucune, toujours active |
| `CUDA_DEVICE_MAX_CONNECTIONS=32` (LN-05) | une file matérielle par flux au lieu de 8 partagées | poser une autre valeur dans l'environnement |
| VA-API non essayé sur NVIDIA propriétaire (LN-14) | pas d'échec à chaque ouverture | aucune |
| Smooth Motion grisé sans RIFE (LN-15) | évite la lecture calée | |
| Resizable BAR journalisé et affiché au premier lancement (LN-09) | « Resizable BAR actif, 12 Go » ou avertissement | |
| Télémétrie NVML dans l'OSD, touche Tab (LN-10) | horloges, P-state, puissance, température, PCIe, écrêtage ; avertit si les horloges sont basses sous charge | |
| Versions minimales du pilote : RTX HDR 550, Smooth Motion 580.82.07 | avertissement au journal | |

## Image : composition finale (LN-06, LN-08)

Onglet HDR/Image des paramètres, « Composition finale » :
- **Filtre vers la fenêtre** : défaut de libplacebo, EWA Lanczos, EWA Lanczos net. Utile quand la
  chaîne ne produit pas la taille de l'écran (par exemple 1440p vers une dalle 4K).
- **Débander les aplats et dégradés** : adoucit les marches des aplats d'anime.
Sans effet sur l'export. Clés `play.toml` : `[picture] scaler`, `deband`.

## DistilDRBA en 4K : flot à l'échelle 0,5

L'auteur de DistilDRBA conseille un flot de 0,5 au lieu de 1 en 4K. Le lecteur prend
`distil_drba_v2_lite_s05.onnx` (plans remplis à un multiple de 128) dès que la hauteur de travail
atteint 1440p, et le graphe entier sinon ; l'un sert de repli à l'autre. Aucune bascule dans
l’environnement (règle du projet) : le seuil est dans `drba_half_flow`. **À mesurer** : PSNR/SSIM entre les deux
sur un extrait en 2160p, et temps par paire (`--stats`).

Les graphes GMFSS Fortuna (base, union, avec demi-flot) et le demi-flot de DRBA s'exportent par le
workflow `Exporter les modèles d'interpolation` (Actions), qui peut les joindre à la release
`models-interp-1` ; `tools/release/fetch-models.sh` les ajoute alors aux paquets.

## GMFSS : performance ou qualité, toujours au flot à demi-échelle

Mesuré sur la 4070 Ti (TensorRT 11.3, 1280×768, graphes « paire » et « fusion » ensemble) : le flot à
demi-échelle (`_s05`) vaut le flot entier ou le dépasse, pour moitié moins de calcul. Le flot entier n'est
donc plus exporté et le lecteur ne le prend plus qu'en repli, à toutes les hauteurs de travail.

| Mode | Base | Union | Écart à l'original (pire pixel / moyenne / 99,9 %, sur 1) |
|---|---|---|---|
| Performance (GMFlow FP16, défaut) | 24,1 ms | 23,3 ms | base 0,0239 / 0,0004 / 0,0059 |
| Qualité (GMFlow FP32, paire `_s05q`) | 30,9 ms | 33,1 ms | base 0,0056 / 0,0002 / 0,0013 |
| Plein flot FP16 (ancien défaut) | 47,1 ms | 46,1 ms | base 0,0055 / 0,0003 / 0,0023 |

Temps mesurés sur un GPU au calme, sans option de profil (graphe « paire » + graphe « fusion ») : paire
7,8 ms (base) et 8,1 ms (union) en performance, 14,6 et 17,9 ms en qualité, 30,8 et 30,9 ms en plein
flot ; fusion 16,3 ms (base) et 15,2 ms (union), la même dans les trois modes. Un premier relevé, fait
pendant que d'autres tâches chargeaient le GPU, gonflait ces temps de 20 à 50 % : un graphe « paire »
identique mesurait 14,0 ms au lieu de 7,8. Mesurer GMFSS demande un GPU libre. L'écart à l'original est
mesuré sur GPU avec PyTorch, indépendant de ces temps.

Le réglage est dans l'onglet d'interpolation (`gmfss_precision` de `play.toml`, section `interpolation` :
`performance`, `balanced` ou `quality`). Les trois crans de précision sont Performance (FP8), Équilibre (FP16,
l'ancien « Performance » de GMFSS) et Qualité (FP32) ; les fichiers d'avant qui portent `gmfss_quality = true`
se lisent comme Qualité. Le cran Performance demande `gmfss_fortuna_{base,union}_s05fp8_pair.onnx` (GMFlow
quantifié en FP8, nœuds Q/DQ) et une RTX 40 ou plus récente ; sans eux ou si TensorRT les refuse, l'Équilibre
reste. Les mêmes crans existent pour SPAN (`image.span_precision`, variantes `-fp8-` et `-fp32-` écrites par
`tools/span/quantize_span.py --precision`), RIFE et DistilDRBA (`interpolation.motion_precision`, graphes
`_fp8.onnx` et `_fp32.onnx` à côté du modèle). Mesure à retenir : un SPAN Balanced aux Q/DQ posés à la main
tourne 15 % plus lentement en FP8 (19,90 ms) qu'en FP16 (17,34 ms) sur une 4070 Ti ; le script de
quantification refuse donc d'écrire une variante FP8 qui ne va pas au moins 5 % plus vite.

Le mode qualité demande `gmfss_fortuna_{base,union}_s05q_pair.onnx` (export_gmfss.py `--scale 0.5 --precision
flownet32`), sa fusion étant celle de `_s05`. Sans cette paire, le mode performance reste.
Le contrôle de l'export juge la moyenne et le 99,9 % de l'écart à l'original, pas le pire pixel d'une petite
image : à 256×256 il valait 0,097 sur un poste et 0,127 en CI même avec tout en FP32 (les sorties du graphe
sont arrondies en FP16).

## Paquet TensorRT indépendant de l'étiquette (depuis la 0.6)

Le paquet TensorRT (notre module et trtexec) ne change presque jamais d'une release à l'autre ; le joindre à chaque étiquette,
puis relancer Release Linux pour que son index entre dans l'`engines.lock`, coûtait plus de 30 minutes par release (v0.5.1).
Il est donc versionné par **l'empreinte de ses entrées** (`tools/release/trt-fingerprint.sh` : le code que le module lie,
les définitions CMake, la recette des dépendances et de l'image, l'empaquetage, l'index NVIDIA ; pas `src/app`, `src/audio`,
`src/render`, la documentation ni les tests ; pas non plus le `CMakeLists.txt` racine, dont la VERSION change à chaque release) et
vit dans la release fixe `trt-<empreinte>` du dépôt des releases.

```
tools/release/trt-package.sh --check   # dit si le paquet de l'empreinte courante existe (code 0) ou manque (code 3)
tools/release/trt-package.sh           # le construit (préflight Docker) et le joint à trt-<empreinte>, une seule fois
```

`release-linux.yml` lit `engines-nvidia.lock` dans `trt-<empreinte>` pour n'importe quelle étiquette : si le paquet existe, la
release se fait en une passe, sans rien à relancer ; s'il manque, le workflow avertit et dit la commande à lancer sur le poste NVIDIA
(puis relancer le workflow). L'image Docker de préflight garde ses couches : seule une modification de `versions.env` ou de
`tools/deps/linux-from-source.sh` la reconstruit (c'est ce qui a coûté ~30 minutes à la 0.5.1), et ces deux fichiers font
justement partie de l'empreinte : le paquet et l'image changent ensemble, jamais l'un sans l'autre. L'ancienne méthode (paquet
joint à l'étiquette elle-même) reste lue en second.

## Joindre le moteur TensorRT à une étiquette (ancienne méthode, poste NVIDIA)

Le paquet TensorRT d'une release ne porte que notre module et trtexec ; libnvinfer, le lecteur ONNX, NVRTC
et les ressources du constructeur de la carte se tirent de chez NVIDIA (`tools/release/nvidia-vendor/`).
Il se construit sur un poste NVIDIA et se joint à la release avant l'étiquette ; la CI fusionne ensuite son
index dans l'`engines.lock` de la release :

```
# Depuis la 0.5, le paquet se construit dans le conteneur de préflight (image reconstruite d'abord, contrôle de portabilité intact) :
#   tools/release/preflight/nvidia-package.sh <worktree à l'étiquette> v0.3.0 <sortie>
# À la main, sur un poste dont les bibliothèques sont celles d'Ubuntu 24.04 :
AJP_WITH_CUDA=ON AJP_TENSORRT_ROOT=$HOME/ajp-sdk/tensorrt AJP_PACKAGE_NVIDIA=1 AJP_DEPS_PREFIX=... \
AJP_RELEASE_REPO=<dépôt des releases> AJP_RELEASE_TAG=v0.3.0 tools/release/package-linux.sh <dossier>
tools/release/attach-nvidia-linux.sh <dossier> v0.3.0               # contrôle seulement
tools/release/attach-nvidia-linux.sh <dossier> v0.3.0 --publish     # crée la release si besoin et joint
git tag v0.3.0 && git push origin v0.3.0                            # la CI fusionne et joint le reste
```

Le script refuse une archive qui contient des fichiers NVIDIA, un index qui ne nomme pas cette archive à cette
étiquette et dans ce dépôt (hachage et taille compris), ou sans `[[vendor_file]]`. Le paquet ne doit pas
changer entre la construction et l'étiquette : son SHA-256 est dans l'index. Le test caché
`ajp_app_tests "[.package]"` (`AJP_PACKAGE_DIR=<dossier>`) installe le paquet avec le vrai code du lecteur,
fichiers de NVIDIA tirés pour de bon (~0,74 Go pour une carte sm89).

## Réglages de départ pour la 4070 Ti

| Réglage | Valeur |
|---|---|
| Moteur | TensorRT (build `linux-prefix-cuda-machine`), Vulkan en repli |
| Profil au premier lancement | 1 (1440p, RIFE ×3) |
| Facteur RIFE | ×3 ; ×4 en scènes calmes |
| Hauteur de travail | 720p |
| HDR | Maxine, puis comparaison à la route native + grade Éclatant |
| Vrai noir | Auto |
| Cadence fixe | à activer si l'OLED scintille en VRR |
| Smooth Motion | désactivé |
| BIOS | Above 4G Decoding et Resizable BAR actifs |
| Pendant la lecture | fermer le navigateur qui décode en NVDEC ; mode « performances maximales » |

## Mesures à faire (audit, §Mesures)

E1 : état du BAR (le lecteur l'affiche maintenant). E2 : Vulkan contre TensorRT, bureau au repos
(`--stats`). E3 : plafond ×3/×4, navigateur fermé. E4 : horloges pendant 2 min de ×3 (la ligne GPU
de l'OSD le montre). E5 : trace Nsight Systems de 20 s en ×3, avec `CUDA_DEVICE_MAX_CONNECTIONS` par
défaut puis à 8. E6 : SPAN INT8 contre FP16.

## Pas encore fait

LN-02/03/12/20 sont des mesures ou des modèles (INT8 de RIFE, flux optique matériel). LN-11 (famille
de files de calcul dédiée), LN-16 (préréglages Vulkan pour d'autres modèles) et LN-17 (présentation
horodatée) attendent les mesures E2 et E5. LN-18 (blanc de référence de KWin) demande
`color-management-v1` ou libdisplay-info. La vitesse de lecture demande un étirement audio sans
changement de hauteur (le régulateur de dérive est borné à quelques millièmes).

## Contre-audit NVIDIA Linux (N1 à N14)

Traités : N1 (cudaMemGetInfo et NVML lus à 2 Hz par un fil à part, plus sur le fil d'affichage), N6
(le partage du contexte primaire ne teste que le mode d'ordonnancement), N7 (`hold()` attend la plus
ancienne image tenue), N10 (l'avertissement Resizable BAR ne sort plus sous TensorRT).

Non traités, à lire sur la carte ou trop risqués sans compilation CUDA : N2 (`cudaDeviceSynchronize`
et `cudaFree` en cours de lecture : mesure M4 d'abord), N3 (TrueHDR en 8 bits : mire de dégradés), N4
(Spline36 tronquée à 32 prises : seulement un avertissement), N5 (drapeau d'allocation dédiée de
l'import), N8 (seuil PSNR de l'autotuner), N9, N11 à N14.
