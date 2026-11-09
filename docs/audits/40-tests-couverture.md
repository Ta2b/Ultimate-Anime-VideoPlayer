# Audit 40. Tests et couverture

**Audit transverse.** Il traverse les zones au lieu de les découper. Cette fiche se suffit à elle-même : elle donne les points d'entrée dans le code et le numéro de la zone de chacun.

## Pourquoi

Savoir ce qui est vraiment protégé par les tests, et ce qui ne l'est pas, avant de corriger à grande échelle.

## Méthode

1. Rattacher chaque test à ce qu'il protège ; relever les chemins critiques sans test (boucle du lecteur, bout en bout, GPU).
2. Relever les tests fragiles : `sleep`, horloge réelle, textes exacts.
3. Relever ce que la CI exécute sur chaque système, et ce qu'elle n'exécute jamais.
4. Proposer un plan de tests, du plus rentable au moins rentable.

## À couvrir

- Tests qui ne testent pas ce qu'ils annoncent.
- Aucun test GPU en CI.
- Résultats ignorés non vérifiés.

## Points d'entrée dans le code

Relevés au commit audité. Partir de ces lignes et suivre les appels ; si une ligne a bougé, chercher le nom indiqué.

### Ce que la CI exécute

| Point | Où | Zone |
|---|---|---|
| CI principale | `.github/workflows/ci.yml` | [28](28-ci-publication.md) |
| CI dans le cloud | `.github/workflows/cloud.yml` | [28](28-ci-publication.md) |
| déclaration des tests | `tests/CMakeLists.txt` | [28](28-ci-publication.md) |

### Tests par zone

Fichiers de tests rattachés à chaque zone de la cible. Une zone à 0 n'a aucun test unitaire.

| Zone | Fichiers de tests |
|---|---|
| [01. TensorRT : exécution, voies, anneau et mémoire](01-tensorrt-execution.md) | 2 |
| [02. SPAN FP8 : noyaux CUTLASS et dérivation](02-span-fp8.md) | 9 |
| [03. Interpolation fusionnée sous TensorRT (RIFE, DistilDRBA, warp)](03-interpolation-tensorrt.md) | 0 |
| [04. Analyse et découpe des modèles ONNX](04-onnx-modeles.md) | 8 |
| [05. Construction des moteurs, cache et trtexec](05-construction-moteurs.md) | 9 |
| [06. Installation des moteurs et téléchargements](06-installation-moteurs.md) | 4 |
| [07. Calibration, niveaux et mode Auto](07-calibration-niveaux.md) | 12 |
| [08. Boucle du lecteur et ouverture de la chaîne](08-boucle-chaine.md) | 3 |
| [09. Étages d'agrandissement et d'interpolation](09-etages-traitement.md) | 19 |
| [10. Rendu et présentation par libplacebo](10-presentation-libplacebo.md) | 4 |
| [11. Écran et matériel : EDID, VRR, HDR du système, télémétrie](11-ecran-materiel.md) | 4 |
| [12. Cadence et horloges](12-cadence-horloges.md) | 9 |
| [13. Session de lecture : seek, épisodes et reprise](13-session-lecture.md) | 13 |
| [14. Sources, démultiplexage et erreurs d'E/S](14-sources-demux.md) | 11 |
| [15. Décodage, matériel, formats et couleurs](15-decodage-formats.md) | 10 |
| [16. Audio](16-audio.md) | 15 |
| [17. Sous-titres](17-sous-titres.md) | 11 |
| [18. Démarrage et arrêt](18-demarrage-arret.md) | 6 |
| [19. Intégration au système](19-integration-systeme.md) | 4 |
| [20. Socle commun (core)](20-socle-core.md) | 12 |
| [21. Frontière des moteurs (interface et ABI)](21-frontiere-moteurs.md) | 13 |
| [22. Réglages : modèle, application et persistance](22-reglages-modele.md) | 5 |
| [23. Réglages : le panneau](23-reglages-panneau.md) | 0 |
| [24. Interface : rendu, contrôles et messages](24-interface-rendu.md) | 4 |
| [25. Assistant de premier lancement](25-assistant.md) | 2 |
| [26. Mise à jour automatique et signature](26-mise-a-jour.md) | 4 |
| [27. Build, dépendances et ports](27-build-dependances.md) | 0 |
| [28. CI, publication et signature](28-ci-publication.md) | 19 |
| [41. HDR, grade et looks](41-hdr-grade.md) | 10 |
| [42. Analyse des plans, coupes et pré-analyse](42-plans-coupes.md) | 5 |
| [43. Textes et traductions](43-textes-traductions.md) | 2 |
| [44. Représentation intermédiaire et import ONNX](44-ir-onnx.md) | 16 |
| [45. Moteur Vulkan : calcul, noyaux et shaders](45-backend-vulkan.md) | 57 |
| [46. Outils de recherche, modèles et données livrées](46-outils-donnees.md) | 0 |

## Défauts déjà connus de ce type

Relevés par le [contre-audit du 08/11](https://claude.ai/artifact/4rfGFb3P7zNiyHiwkfcnUU). Les revérifier, et chercher les autres.

| ID | Gravité | Défaut |
|---|---|---|
| B3 | Bloquant | Le chemin principal n'a jamais tourné avant la livraison : aucun test GPU en CI, installation TensorRT Windows non faite, mise à jour Windows jamais testée, aucune mesure AMD. |
| QUA-04 | Majeur | Les résultats ignorés ne sont vérifiés ni sous Windows, ni hors des PR. Les sites relevés le 06/11 sont intacts. |
| SEC-22 | Mineur | Les tests de révocation et de bout en bout ne testent pas ce qu'ils annoncent. L'application n'est jamais testée sous Windows. |

## Zones traversées

Les fiches de ces zones donnent la liste complète de leurs fichiers et leurs défauts connus. Les lire pour le contexte, sans refaire leur audit : un défaut local trouvé en chemin va dans « Renvois ».

- [01. TensorRT : exécution, voies, anneau et mémoire](01-tensorrt-execution.md)
- [02. SPAN FP8 : noyaux CUTLASS et dérivation](02-span-fp8.md)
- [03. Interpolation fusionnée sous TensorRT (RIFE, DistilDRBA, warp)](03-interpolation-tensorrt.md)
- [04. Analyse et découpe des modèles ONNX](04-onnx-modeles.md)
- [05. Construction des moteurs, cache et trtexec](05-construction-moteurs.md)
- [06. Installation des moteurs et téléchargements](06-installation-moteurs.md)
- [07. Calibration, niveaux et mode Auto](07-calibration-niveaux.md)
- [08. Boucle du lecteur et ouverture de la chaîne](08-boucle-chaine.md)
- [09. Étages d'agrandissement et d'interpolation](09-etages-traitement.md)
- [10. Rendu et présentation par libplacebo](10-presentation-libplacebo.md)
- [11. Écran et matériel : EDID, VRR, HDR du système, télémétrie](11-ecran-materiel.md)
- [12. Cadence et horloges](12-cadence-horloges.md)
- [13. Session de lecture : seek, épisodes et reprise](13-session-lecture.md)
- [14. Sources, démultiplexage et erreurs d'E/S](14-sources-demux.md)
- [15. Décodage, matériel, formats et couleurs](15-decodage-formats.md)
- [16. Audio](16-audio.md)
- [17. Sous-titres](17-sous-titres.md)
- [18. Démarrage et arrêt](18-demarrage-arret.md)
- [19. Intégration au système](19-integration-systeme.md)
- [20. Socle commun (core)](20-socle-core.md)
- [21. Frontière des moteurs (interface et ABI)](21-frontiere-moteurs.md)
- [22. Réglages : modèle, application et persistance](22-reglages-modele.md)
- [23. Réglages : le panneau](23-reglages-panneau.md)
- [24. Interface : rendu, contrôles et messages](24-interface-rendu.md)
- [25. Assistant de premier lancement](25-assistant.md)
- [26. Mise à jour automatique et signature](26-mise-a-jour.md)
- [27. Build, dépendances et ports](27-build-dependances.md)
- [28. CI, publication et signature](28-ci-publication.md)
- [41. HDR, grade et looks](41-hdr-grade.md)
- [42. Analyse des plans, coupes et pré-analyse](42-plans-coupes.md)
- [43. Textes et traductions](43-textes-traductions.md)
- [44. Représentation intermédiaire et import ONNX](44-ir-onnx.md)
- [45. Moteur Vulkan : calcul, noyaux et shaders](45-backend-vulkan.md)
- [46. Outils de recherche, modèles et données livrées](46-outils-donnees.md)
