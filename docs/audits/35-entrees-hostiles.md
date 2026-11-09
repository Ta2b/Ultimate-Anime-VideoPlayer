# Audit 35. Entrées non sûres

**Audit transverse.** Il traverse les zones au lieu de les découper. Cette fiche se suffit à elle-même : elle donne les points d'entrée dans le code et le numéro de la zone de chacun.

## Pourquoi

Tout ce qui entre dans le lecteur depuis l'extérieur peut être piégé. Cet audit fait le modèle de menace complet et vérifie chaque point d'entrée.

## Méthode

1. Lister chaque entrée : fichiers vidéo, sous-titres, polices jointes, modèles ONNX, LUT, fichiers de réglages, manifeste et archives de mise à jour, index des moteurs, arguments, messages de l'instance unique, variables d'environnement.
2. Pour chacune : qui la lit, quelles bornes, quel fuzzing.
3. Classer par risque réel sur la cible.

## À couvrir

- Bornes de taille et de récursion.
- Chemins tirés de données externes.
- Décompression sans plafond.
- Couverture du fuzzing.

## Points d'entrée dans le code

Relevés au commit audité. Partir de ces lignes et suivre les appels ; si une ligne a bougé, chercher le nom indiqué.

### Points d'entrée des données externes

| Point | Où | Zone |
|---|---|---|
| fichier vidéo | `src/media/demuxer.cpp:160` | [14](14-sources-demux.md) |
| polices jointes des sous-titres | `src/subs/ass_library.cpp:59` | [17](17-sous-titres.md) |
| LUT .cube | `src/app/play/look_catalog.cpp:56` | [41](41-hdr-grade.md) |
| modèles ONNX | `src/prepare/onnx_scan.cpp:33` | [04](04-onnx-modeles.md) |
| réglages lus avec tolérance | `src/app/play/settings_toml_lenient.cpp` | [22](22-reglages-modele.md) |
| réglages de lecture | `src/app/play/settings_toml.cpp:146` | [22](22-reglages-modele.md) |
| table des cartes | `src/app/play/gpu_tier.cpp:63` | [07](07-calibration-niveaux.md) |
| manifeste de mise à jour | `src/core/update_manifest.cpp` | [26](26-mise-a-jour.md) |
| archives décompressées | `src/core/inflate.cpp:28` | [06](06-installation-moteurs.md) |
| catalogue des moteurs | `src/app/play/engine_install.cpp:31` | [06](06-installation-moteurs.md) |
| message d'un second lancement | `src/app/play/single_instance.cpp:96` | [19](19-integration-systeme.md) |
| fichiers glissés dans la fenêtre | `src/app/play/home_screen.cpp:202` | [24](24-interface-rendu.md) |
| variables d'environnement | `src/app/play/updater.cpp:482` | [26](26-mise-a-jour.md) |

## Défauts déjà connus de ce type

Relevés par le [contre-audit du 08/11](https://claude.ai/artifact/4rfGFb3P7zNiyHiwkfcnUU). Les revérifier, et chercher les autres.

| ID | Gravité | Défaut |
|---|---|---|
| MED-14 | Mineur | `duration_to_us` peut rendre `AV_NOPTS_VALUE`, et `chain_dimension(1)` vaut 0. Fichier hostile seulement. |
| SEC-10 | Mineur | La version et le nom d'archive tirés du manifeste signé ne sont pas assainis. Une clé compromise permet d'écrire hors de `updates/`. |
| SEC-11 | Mineur | Sous Windows, `\Users\x\…` et `D:x` passent la garde de `.ajp-files`. Une liste altérée fait supprimer des fichiers hors de l'installation. |
| SEC-12 | Mineur | Sans `staged_sha256`, l'archive est extraite sans vérification. Elle peut aussi changer entre la vérification et l'extraction. |
| SEC-14 | Mineur | La décompression des fichiers NVIDIA n'a pas de plafond avant le contrôle du SHA. Le `.part` est repris par son nom (voir TRT-10). |
| SEC-15 | Mineur | Le canal nommé Windows a un nom prévisible et l'ACL par défaut. Sur un poste partagé, un autre utilisateur capte les vidéos ouvertes ou bloque le lecteur (voir WIN-06). |
| SEC-21 | Mineur | La graine ONNX fait 16 octets, le corpus est jeté chaque nuit, et `scan_onnx` n'est pas fuzzé. CodeQL tourne sans construction. |

## Zones traversées

Les fiches de ces zones donnent la liste complète de leurs fichiers et leurs défauts connus. Les lire pour le contexte, sans refaire leur audit : un défaut local trouvé en chemin va dans « Renvois ».

- [04. Analyse et découpe des modèles ONNX](04-onnx-modeles.md)
- [06. Installation des moteurs et téléchargements](06-installation-moteurs.md)
- [07. Calibration, niveaux et mode Auto](07-calibration-niveaux.md)
- [14. Sources, démultiplexage et erreurs d'E/S](14-sources-demux.md)
- [17. Sous-titres](17-sous-titres.md)
- [19. Intégration au système](19-integration-systeme.md)
- [22. Réglages : modèle, application et persistance](22-reglages-modele.md)
- [24. Interface : rendu, contrôles et messages](24-interface-rendu.md)
- [26. Mise à jour automatique et signature](26-mise-a-jour.md)
- [41. HDR, grade et looks](41-hdr-grade.md)
