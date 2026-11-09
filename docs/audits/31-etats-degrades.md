# Audit 31. États dégradés et reprise sur erreur

**Audit transverse.** Il traverse les zones au lieu de les découper. Cette fiche se suffit à elle-même : elle donne les points d'entrée dans le code et le numéro de la zone de chacun.

## Pourquoi

Ce que vit l'utilisateur quand quelque chose tourne mal. L'audit croise chaque panne avec chaque périmètre et vérifie que le lecteur se rétablit ou explique clairement.

## Méthode

1. Construire une matrice : pannes en lignes, périmètres en colonnes.
2. Pour chaque case : comportement réel dans le code, message montré, état laissé sur le disque.
3. Proposer pour chaque cas le comportement attendu et le test qui le prouve (injection d'erreurs).

## À couvrir

- Perte du GPU : TDR sous Windows, Xid sous Linux, eGPU débranché.
- Mémoire vidéo pleine, mémoire vive pleine.
- Disque plein, dossier en lecture seule, antivirus qui tient un fichier.
- Réseau coupé : partage SMB, téléchargement, vérification de mise à jour.
- Fichier vidéo corrompu ou tronqué, sous-titres illisibles.
- Pilote NVIDIA trop ancien, TensorRT absent ou abîmé, DLL manquante.
- Écran débranché, appareil audio débranché, veille et hibernation.

## Points d'entrée dans le code

Relevés au commit audité. Partir de ces lignes et suivre les appels ; si une ligne a bougé, chercher le nom indiqué.

### Où les pannes arrivent

| Point | Où | Zone |
|---|---|---|
| erreurs CUDA | `src/gpu/cuda/backend/cuda_warn.h` | [01](01-tensorrt-execution.md) |
| nouvelles tentatives de la source | `src/media/source_loop.cpp:110` | [14](14-sources-demux.md) |
| lecture interrompue | `src/media/demuxer.cpp:208` | [14](14-sources-demux.md) |
| échec d'un étage d'anneau | `src/app/play/ring_stage.cpp:106` | [09](09-etages-traitement.md) |
| erreur montrée à l'utilisateur | `src/app/play/visible_error.h` | [24](24-interface-rendu.md) |
| fichier mis de côté | `src/app/play/player.cpp:77` | [18](18-demarrage-arret.md) |
| écriture atomique | `src/core/file_sync.cpp:63` | [20](20-socle-core.md) |
| verrou de cache | `src/prepare/key_lock.cpp:80` | [05](05-construction-moteurs.md) |
| cache intact ? | `src/prepare/cache.cpp:48` | [05](05-construction-moteurs.md) |

### Tous les points qui détectent une perte de l'appareil

Peu de lignes : vérifier que chaque chemin de rendu et de calcul de la cible détecte la perte du GPU. (7 lignes, zones de la cible seulement.)

| Où | Zone | Code |
|---|---|---|
| `src/gpu/cuda/backend/cuda_status.h:17` | [01](01-tensorrt-execution.md) | `case cudaErrorIllegalAddress:` |
| `src/gpu/cuda/backend/cuda_status.h:23` | [01](01-tensorrt-execution.md) | `case cudaErrorLaunchFailure:` |
| `src/gpu/vulkan/compute/context.h:204` | [45](45-backend-vulkan.md) | `if (result == VK_ERROR_DEVICE_LOST) mark_lost();` |
| `src/gpu/vulkan/compute/queue.cpp:117` | [45](45-backend-vulkan.md) | `if (result == VK_ERROR_DEVICE_LOST) mark_lost();` |
| `src/gpu/vulkan/compute/refuse.cpp:23` | [45](45-backend-vulkan.md) | `case VK_ERROR_DEVICE_LOST:` |
| `src/gpu/vulkan/compute/refuse.cpp:24` | [45](45-backend-vulkan.md) | `return "VK_ERROR_DEVICE_LOST";` |
| `src/gpu/vulkan/compute/refuse.cpp:57` | [45](45-backend-vulkan.md) | `: result == VK_ERROR_DEVICE_LOST ? RefusalCode::DeviceLost` |

## Défauts déjà connus de ce type

Relevés par le [contre-audit du 08/11](https://claude.ai/artifact/4rfGFb3P7zNiyHiwkfcnUU). Les revérifier, et chercher les autres.

| ID | Gravité | Défaut |
|---|---|---|
| GPU-06 | Majeur | Une perte d'appareil ou de surface côté présentation fige l'image sans message. Le journal reçoit 250 à 1 000 lignes par seconde. |
| MED-01 | Majeur | Une erreur d'E/S de plus de 1,55 s devient une fin de flux, et l'épisode suivant démarre. La notice affiche « Playback goes on after an error ». |
| QUA-07 | Majeur | Sous Windows, l'échec d'écriture de `update.toml` (antivirus) est ignoré. Le lancement compte comme un plantage, et la version peut être retirée après 3 lancements. |
| GPU-17 | Mineur | Un GPU bloqué sans être perdu n'est pas détecté. La fermeture peut rester bloquée dans `pl_gpu_finish`. |
| MED-06 | Mineur | Un seek pendant une panne d'E/S peut laisser la lecture figée, sans fin ni délai. Le rognage audio peut viser le seek manqué. |
| MED-08 | Mineur | Au retrait d'un appareil, son délai reste appliqué aux haut-parleurs. Sans sortie par défaut, chaque bloc échoue au journal et rien n'est retenté. |
| TRT-03 | Mineur | Sur `AbiMismatch`, le module est retéléchargé et réinstallé à chaque lancement, sans jamais réussir. Démarrage lent et environ 1 Go rehaché. |
| TRT-05 | Mineur | Le délai de `trtexec` (20 min) égale la plus longue construction mesurée. Sur un portable lent, la construction échoue et recommence à chaque ouverture. |

## Zones traversées

Les fiches de ces zones donnent la liste complète de leurs fichiers et leurs défauts connus. Les lire pour le contexte, sans refaire leur audit : un défaut local trouvé en chemin va dans « Renvois ».

- [01. TensorRT : exécution, voies, anneau et mémoire](01-tensorrt-execution.md)
- [05. Construction des moteurs, cache et trtexec](05-construction-moteurs.md)
- [09. Étages d'agrandissement et d'interpolation](09-etages-traitement.md)
- [14. Sources, démultiplexage et erreurs d'E/S](14-sources-demux.md)
- [18. Démarrage et arrêt](18-demarrage-arret.md)
- [20. Socle commun (core)](20-socle-core.md)
- [24. Interface : rendu, contrôles et messages](24-interface-rendu.md)
- [45. Moteur Vulkan : calcul, noyaux et shaders](45-backend-vulkan.md)
