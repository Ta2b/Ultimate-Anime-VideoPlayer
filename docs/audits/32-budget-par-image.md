# Audit 32. Budget par image sur la cible

**Audit transverse.** Il traverse les zones au lieu de les découper. Cette fiche se suffit à elle-même : elle donne les points d'entrée dans le code et le numéro de la zone de chacun.

## Pourquoi

Le temps de chaque étape, de l'image décodée à l'image affichée, sous TensorRT, pour savoir où va chaque milliseconde et ce qui menace le rythme à 144 ou 165 Hz.

## Méthode

1. Choisir trois configurations de référence avec Tatoub, par exemple 1080p vers 1440p à 165 Hz en ×3, et 1080p vers 2160p à 60 Hz.
2. Pour chaque image, lister les étapes CPU et GPU, leurs threads et leurs points de synchronisation, à partir du code et des mesures de `docs/reports`.
3. Relever tout travail fait à chaque image sur le thread principal : E/S, requêtes système, allocations, formatages.
4. Comparer au budget de l'écran et classer les gains possibles.

## À couvrir

- Copies hôte et envoi des images.
- SPAN, RIFE, warp, anneau.
- Grade HDR, sous-titres, interface rendue à chaque présentation.
- Requêtes système périodiques.
- FP8 non servi par l'Auto.

## Points d'entrée dans le code

Relevés au commit audité. Partir de ces lignes et suivre les appels ; si une ligne a bougé, chercher le nom indiqué.

### Le trajet d'une image

| Point | Où | Zone |
|---|---|---|
| lecture et paquets | `src/media/source_loop.cpp:42` | [14](14-sources-demux.md) |
| envoi des images logicielles | `src/gpu/cuda/backend/host_upload.cpp:82` | [01](01-tensorrt-execution.md) |
| étage de l'anneau | `src/app/play/ring_stage.cpp:51` | [09](09-etages-traitement.md) |
| choix des sorties d'une image | `src/app/play/upscale_stage.cpp:274` | [09](09-etages-traitement.md) |
| production d'une sortie | `src/app/play/upscale_stage.cpp:640` | [09](09-etages-traitement.md) |
| SPAN FP8 | `src/gpu/cuda/backend/tensorrt_span_fp8.cpp:66` | [02](02-span-fp8.md) |
| image reçue par la session | `src/app/play/session.cpp:250` | [13](13-session-lecture.md) |
| boucle de la chaîne | `src/app/play/player_run.cpp:98` | [08](08-boucle-chaine.md) |
| requêtes d'écran | `src/app/play/host_facts.cpp:89` | [11](11-ecran-materiel.md) |
| télémétrie | `src/app/play/gpu_telemetry.cpp:63` | [11](11-ecran-materiel.md) |

## Défauts déjà connus de ce type

Relevés par le [contre-audit du 08/11](https://claude.ai/artifact/4rfGFb3P7zNiyHiwkfcnUU). Les revérifier, et chercher les autres.

| ID | Gravité | Défaut |
|---|---|---|
| WIN-04 | Majeur | Les requêtes d'affichage restent sur le thread principal, en double toutes les 500 ms, et la fabrique DXGI est recréée toutes les 2 s. Majeur à 144-165 Hz (voir GPU-16). |
| GPU-16 | Mineur | SetupDi, registre, DXGI, `QueryDisplayConfig` et NVAPI sont appelés sur le thread de rendu toutes les 500 ms ou 2 s. Majeur à 144-165 Hz si confirmé (voir WIN-04). |
| MED-03 | Mineur | La conversion P0.11 alloue environ 25 Mo par image 4K sans pool, en monothread, même pour les images jetées et dans la pré-analyse. Majeur en 4K si la mesure le confirme. |
| TRT-08 | Mineur | Sur un écran 1080p, la calibration écrit `span_fp8 = true` sans l'avoir jugé et change le modèle. Elle peut aussi mesurer un FP8 non validé. |

## Zones traversées

Les fiches de ces zones donnent la liste complète de leurs fichiers et leurs défauts connus. Les lire pour le contexte, sans refaire leur audit : un défaut local trouvé en chemin va dans « Renvois ».

- [01. TensorRT : exécution, voies, anneau et mémoire](01-tensorrt-execution.md)
- [02. SPAN FP8 : noyaux CUTLASS et dérivation](02-span-fp8.md)
- [08. Boucle du lecteur et ouverture de la chaîne](08-boucle-chaine.md)
- [09. Étages d'agrandissement et d'interpolation](09-etages-traitement.md)
- [11. Écran et matériel : EDID, VRR, HDR du système, télémétrie](11-ecran-materiel.md)
- [13. Session de lecture : seek, épisodes et reprise](13-session-lecture.md)
- [14. Sources, démultiplexage et erreurs d'E/S](14-sources-demux.md)
