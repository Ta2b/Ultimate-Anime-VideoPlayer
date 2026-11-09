# Audit 08. Boucle du lecteur et ouverture de la chaîne

**Audit par zone. Priorité : haute.** 15 fichiers, 3 117 lignes ; tests rattachés : 3 fichiers, 605 lignes.

## Mission

La boucle principale de lecture et tout ce qui ouvre, rouvre ou change la chaîne : choix des moteurs, construction à la volée, réouvertures, statistiques. C'est le code le plus complexe du dépôt (`run_chain`).

## Points à auditer en profondeur

Un point de départ : l'audit doit aussi chercher ce que la liste ne nomme pas.

- Réouvertures : déclencheurs, hystérésis, constructions qu'elles provoquent (GPU-03).
- Chaîne appliquée contre chaîne mesurée (TRT-01). « Annuler et regarder » et `--no-upscale` (TRT-02).
- Réglages pris en compte à la réouverture : tous les champs (UX-01, UX-02).
- Durée de vie des moteurs et de l'anneau à chaque changement : fuites, mémoire vidéo libérée.
- Requêtes système et E/S sur le thread de la boucle (WIN-04, MED-15).
- Complexité de `run_chain` et testabilité (QUA-15).

## Frontières

Dépend de :

- [20. Socle commun (core)](20-socle-core.md) (15 inclusions)
- [24. Interface : rendu, contrôles et messages](24-interface-rendu.md) (15 inclusions)
- [09. Étages d'agrandissement et d'interpolation](09-etages-traitement.md) (15 inclusions)
- [22. Réglages : modèle, application et persistance](22-reglages-modele.md) (14 inclusions)
- [12. Cadence et horloges](12-cadence-horloges.md) (11 inclusions)
- [07. Calibration, niveaux et mode Auto](07-calibration-niveaux.md) (11 inclusions)
- [13. Session de lecture : seek, épisodes et reprise](13-session-lecture.md) (10 inclusions)
- [11. Écran et matériel : EDID, VRR, HDR du système, télémétrie](11-ecran-materiel.md) (10 inclusions)
- [21. Frontière des moteurs (interface et ABI)](21-frontiere-moteurs.md) (9 inclusions)
- [18. Démarrage et arrêt](18-demarrage-arret.md) (8 inclusions)
- [14. Sources, démultiplexage et erreurs d'E/S](14-sources-demux.md) (6 inclusions)
- [43. Textes et traductions](43-textes-traductions.md) (6 inclusions)
- [04. Analyse et découpe des modèles ONNX](04-onnx-modeles.md) (5 inclusions)
- [45. Moteur Vulkan : calcul, noyaux et shaders](45-backend-vulkan.md) (4 inclusions)
- [41. HDR, grade et looks](41-hdr-grade.md) (3 inclusions)
- [10. Rendu et présentation par libplacebo](10-presentation-libplacebo.md) (3 inclusions)
- [25. Assistant de premier lancement](25-assistant.md) (3 inclusions)
- [06. Installation des moteurs et téléchargements](06-installation-moteurs.md) (3 inclusions)
- [16. Audio](16-audio.md) (2 inclusions)
- [42. Analyse des plans, coupes et pré-analyse](42-plans-coupes.md) (2 inclusions)
- [05. Construction des moteurs, cache et trtexec](05-construction-moteurs.md) (2 inclusions)
- [47. Export vidéo](47-export.md) (2 inclusions)
- [15. Décodage, matériel, formats et couleurs](15-decodage-formats.md) (1 inclusion)
- [23. Réglages : le panneau](23-reglages-panneau.md) (1 inclusion)
- [19. Intégration au système](19-integration-systeme.md) (1 inclusion)

Utilisé par :

- [18. Démarrage et arrêt](18-demarrage-arret.md) (7 inclusions)
- [24. Interface : rendu, contrôles et messages](24-interface-rendu.md) (5 inclusions)
- [25. Assistant de premier lancement](25-assistant.md) (4 inclusions)
- [47. Export vidéo](47-export.md) (2 inclusions)
- [07. Calibration, niveaux et mode Auto](07-calibration-niveaux.md) (1 inclusion)
- [41. HDR, grade et looks](41-hdr-grade.md) (1 inclusion)
- [42. Analyse des plans, coupes et pré-analyse](42-plans-coupes.md) (1 inclusion)
- [09. Étages d'agrandissement et d'interpolation](09-etages-traitement.md) (1 inclusion)

Auditer l'interface avec ces zones (ce qui entre, ce qui sort, qui possède quoi, quel thread appelle), pas leur intérieur. Les parcours complets sont couverts par les audits transverses.

## Défauts déjà connus

Relevés par le [contre-audit du 08/11](https://claude.ai/artifact/4rfGFb3P7zNiyHiwkfcnUU). Les revérifier un par un, sans s'y limiter.

| ID | Gravité | Défaut |
|---|---|---|
| GPU-01 | Majeur | Le VRR que NVAPI dit armé est ignoré si l'EDID n'annonce pas de plage VRR ou ne se lit pas. P0.15 ne joue pas sur une partie des écrans G-SYNC. |
| GPU-03 | Majeur | Passer la fenêtre sur un écran d'une autre hauteur rouvre toute la chaîne, sans hystérésis. Sous TensorRT, cela peut lancer une construction de plusieurs minutes. |
| MED-01 | Majeur | Une erreur d'E/S de plus de 1,55 s devient une fin de flux, et l'épisode suivant démarre. La notice affiche « Playback goes on after an error ». |
| TRT-02 | Majeur | « Annuler et regarder » joue la vidéo réduite à la hauteur de travail, sans SPAN ni RIFE. `--no-upscale` a le même défaut. |
| UX-02 | Majeur | L'assistant relancé pendant une lecture écrit `play.toml`, puis la chaîne rouvre avec les anciens réglages et les réécrit. Langue, mises à jour et chaîne calibrée sont perdues. |
| UX-13 | Majeur | « Report a problem » promet un journal sans dossiers, mais les chemins complets hors du dossier personnel y restent (voir WIN-08). |
| GPU-02 | Mineur | Cache VRR global de 250 ms, et la branche plein écran pose `vrr_armed` sans mettre à jour `vrr_watch`. Un VRR désengagé peut rester vu armé. |
| GPU-05 | Mineur | Le plancher VRR n'est pas recalculé au changement d'écran si l'état armé ne change pas. |
| GPU-13 | Mineur | La bascule HDR du système est suivie avec jusqu'à 2 s de retard, alors que l'événement SDL est déjà reçu. |
| MED-07 | Mineur | L'export grave la compensation de latence de l'appareil, par exemple +200 ms en Bluetooth. Le fichier est désynchronisé sur tout autre appareil. Mineur à Majeur selon le rapport. |
| MED-15 | Mineur | `Pipeline::open`, la recherche des sous-titres externes et `sibling_episode` tournent sur le thread de la fenêtre. La fenêtre se fige sur un NAS lent. |
| UX-08 | Mineur | Échap ferme tout le panneau et jette les changements, même quand on veut fermer une liste. |
| UX-20 | Mineur | Des textes sont formatés à l'ouverture de la chaîne et restent dans l'ancienne langue jusqu'à la vidéo suivante. |

## Fichiers

| Fichier | Lignes |
|---|---|
| `src/app/play/chain.cpp` | 439 |
| `src/app/play/chain.h` | 177 |
| `src/app/play/chain_setup.cpp` | 529 |
| `src/app/play/chain_setup.h` | 133 |
| `src/app/play/chain_stats.cpp` | 222 |
| `src/app/play/chain_stats.h` | 85 |
| `src/app/play/engine_build.cpp` | 71 |
| `src/app/play/engine_build.h` | 59 |
| `src/app/play/engine_switch.cpp` | 63 |
| `src/app/play/engine_switch.h` | 32 |
| `src/app/play/free_engines.h` | 37 |
| `src/app/play/player_engines.cpp` | 308 |
| `src/app/play/player_engines.h` | 169 |
| `src/app/play/player_run.cpp` | 753 |
| `src/app/play/player_run.h` | 40 |

## Tests rattachés

- `tests/unit/app/chain_setup_test.cpp`
- `tests/unit/app/engine_switch_test.cpp`
- `tests/unit/core/free_test.cpp`

## Hors de cet audit

Tout fichier non listé ci-dessus. Un défaut trouvé ailleurs va dans « Renvois » du rapport, avec le numéro de l'audit concerné.
