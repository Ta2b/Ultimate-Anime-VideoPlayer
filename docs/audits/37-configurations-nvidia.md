# Audit 37. Configurations NVIDIA variées

**Audit transverse.** Il traverse les zones au lieu de les découper. Cette fiche se suffit à elle-même : elle donne les points d'entrée dans le code et le numéro de la zone de chacun.

## Pourquoi

La cible n'est pas une seule machine. Cet audit vérifie chaque décision que le code prend selon le matériel et le système.

## Méthode

1. Lister chaque décision prise selon la carte, la mémoire, le pilote, l'écran ou le système : table des cartes, détection de l'architecture, seuils de mémoire, choix des voies, niveaux.
2. Pour chacune, dérouler les configurations de la cible convenues avec Tatoub.

## À couvrir

- RTX 20, 30, 40 et 50 ; portables avec Optimus ; cartes de 6 à 8 Go ; deux GPU.
- Pilotes NVIDIA, du minimum exigé au plus récent, `nvidia-open` sous CachyOS.
- Écrans de 60 à 240 Hz, VRR ou non, HDR ou non, un ou plusieurs.
- Versions de Windows 11 ; noyaux et bureaux de CachyOS.

## Points d'entrée dans le code

Relevés au commit audité. Partir de ces lignes et suivre les appels ; si une ligne a bougé, chercher le nom indiqué.

### Décisions prises selon le matériel

| Point | Où | Zone |
|---|---|---|
| niveau estimé de la carte | `src/app/play/gpu_tier.cpp:102` | [07](07-calibration-niveaux.md) |
| table des cartes livrée | `data/gpu_tiers.toml` | [07](07-calibration-niveaux.md) |
| architecture CUDA d'après le nom | `src/app/play/engine_install.cpp:97` | [06](06-installation-moteurs.md) |
| architecture détectée | `src/app/play/engine_install.cpp:157` | [06](06-installation-moteurs.md) |
| appel de nvidia-smi | `src/app/play/engine_install.cpp:128` | [06](06-installation-moteurs.md) |
| version du pilote | `src/app/play/first_run.cpp:220` | [25](25-assistant.md) |
| pilote trop ancien | `src/app/play/gpu_telemetry.cpp:129` | [11](11-ecran-materiel.md) |
| nombre de voies TensorRT | `src/gpu/cuda/backend/tensorrt_lanes.cpp:23` | [01](01-tensorrt-execution.md) |
| choix du niveau par l'Auto | `src/app/play/auto_config.cpp:69` | [07](07-calibration-niveaux.md) |
| VRR armé | `src/app/play/window_vrr.cpp:101` | [11](11-ecran-materiel.md) |

## Défauts déjà connus de ce type

Relevés par le [contre-audit du 08/11](https://claude.ai/artifact/4rfGFb3P7zNiyHiwkfcnUU). Les revérifier, et chercher les autres.

| ID | Gravité | Défaut |
|---|---|---|
| GPU-01 | Majeur | Le VRR que NVAPI dit armé est ignoré si l'EDID n'annonce pas de plage VRR ou ne se lit pas. P0.15 ne joue pas sur une partie des écrans G-SYNC. |
| GPU-05 | Mineur | Le plancher VRR n'est pas recalculé au changement d'écran si l'état armé ne change pas. |
| GPU-18 | Mineur | Sous X11 avec le pilote NVIDIA, l'absence de `VRR_ENABLED` donne « non armé » au lieu de « inconnu ». |
| TRT-09 | Mineur | Les quatre niveaux ne donnent pas quatre chaînes pour une source de 720p ou moins : Qualité égale Ultra. |
| TRT-17 | Mineur | Défauts groupés : échec de `nvidia-smi` gardé, carte inconnue en « ptx », moteurs reconstruits à chaque pilote sans purge, doc MPEG-4 fausse, largeur en 16/9 sur ultra-large. |

## Zones traversées

Les fiches de ces zones donnent la liste complète de leurs fichiers et leurs défauts connus. Les lire pour le contexte, sans refaire leur audit : un défaut local trouvé en chemin va dans « Renvois ».

- [01. TensorRT : exécution, voies, anneau et mémoire](01-tensorrt-execution.md)
- [06. Installation des moteurs et téléchargements](06-installation-moteurs.md)
- [07. Calibration, niveaux et mode Auto](07-calibration-niveaux.md)
- [11. Écran et matériel : EDID, VRR, HDR du système, télémétrie](11-ecran-materiel.md)
- [25. Assistant de premier lancement](25-assistant.md)
