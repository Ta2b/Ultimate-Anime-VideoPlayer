# NVIDIA sous Linux, jusqu'à la RTX 4070 Ti

Ce guide suit l'audit « NVIDIA Linux » (`/mnt/project-files/audit/audit-nvidia-linux.html`, constats
LN-01 à LN-20). Il dit ce que le lecteur fait seul sur une carte NVIDIA, ce qui se construit sur la
machine, et ce qu'il reste à mesurer.

## Deux voies

- **Vulkan** : livrée à tous. Décodage Vulkan Video, réseaux en shaders (matrices coopératives,
  préréglages Ada pour SPAN 720p et RIFE 4.25 dans `tools/vk/presets/`).
- **TensorRT + CUDA + NVDEC** : la voie de performance maximale, **construite sur la machine**. Les
  paquets NVIDIA ne sont pas publiés (licence NVIDIA) : l'écran de bienvenue le dit et renvoie ici.

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
