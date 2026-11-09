# Audit 29. Parcours utilisateur de bout en bout

**Audit transverse.** Il traverse les zones au lieu de les découper. Cette fiche se suffit à elle-même : elle donne les points d'entrée dans le code et le numéro de la zone de chacun.

## Pourquoi

Les défauts les plus graves du contre-audit sont nés entre deux zones (TRT-01, MED-01, UX-01). Cet audit suit chaque parcours réel d'un bout à l'autre du code, à travers tous les périmètres qu'il traverse.

## Méthode

1. Pour chaque parcours, écrire le chemin d'appels réel, fichier par fichier, avec les threads concernés.
2. À chaque étape, noter l'état gardé, ce que voit l'utilisateur et ce qui se passe s'il interrompt.
3. Vérifier que l'état final est celui que l'utilisateur attend.

## À couvrir

- Premier lancement par double-clic sur un épisode, sous Windows et sous CachyOS.
- Ouverture d'un épisode avec les moteurs TensorRT prêts, puis avec un moteur à construire.
- « Annuler et regarder » pendant une construction.
- Seek avant et arrière, image par image, pause et reprise.
- Changement de réglage en direct, puis d'un réglage qui rouvre la vidéo.
- Changement de piste audio et de sous-titres, délais.
- Fin d'épisode et passage au suivant ; épisode suivant absent.
- Fenêtre déplacée vers un autre écran, plein écran, réduction, HDR du système activé puis coupé.
- Veille puis reprise du PC pendant une lecture.
- Fermeture pendant une construction, une calibration, un téléchargement, une mise à jour.
- Mise à jour proposée, appliquée, puis relancement.

## Points d'entrée dans le code

Relevés au commit audité. Partir de ces lignes et suivre les appels ; si une ligne a bougé, chercher le nom indiqué.

### Lancement et premier lancement

| Point | Où | Zone |
|---|---|---|
| entrée du programme | `src/app/main.cpp:289` | [18](18-demarrage-arret.md) |
| lecture des options | `src/app/play/options.cpp:78` | [18](18-demarrage-arret.md) |
| instance unique | `src/app/play/single_instance.cpp:210` | [19](19-integration-systeme.md) |
| fichiers transmis par un second lancement | `src/app/play/single_instance.cpp:49` | [19](19-integration-systeme.md) |
| étapes de l'assistant | `src/app/play/wizard.cpp:26` | [25](25-assistant.md) |
| réglages produits par l'assistant | `src/app/play/wizard.cpp:116` | [25](25-assistant.md) |
| lecteur | `src/app/play/player.cpp:129` | [18](18-demarrage-arret.md) |

### Ouverture d'un épisode et moteurs

| Point | Où | Zone |
|---|---|---|
| sonde de la vidéo | `src/app/play/player.cpp:121` | [18](18-demarrage-arret.md) |
| boucle de la chaîne | `src/app/play/player_run.cpp:98` | [08](08-boucle-chaine.md) |
| préparation des moteurs | `src/app/play/chain_setup.cpp:160` | [08](08-boucle-chaine.md) |
| construction en arrière-plan | `src/app/play/engine_build.cpp:33` | [08](08-boucle-chaine.md) |
| bouton « Cancel and watch » | `src/app/play/controls_draw.cpp:382` | [24](24-interface-rendu.md) |
| installation des moteurs | `src/app/play/engine_downloads.cpp:40` | [06](06-installation-moteurs.md) |

### Lecture, seek et réglages

| Point | Où | Zone |
|---|---|---|
| commandes de la session | `src/app/play/session.cpp:73` | [13](13-session-lecture.md) |
| seek | `src/app/play/session.cpp:189` | [13](13-session-lecture.md) |
| seek côté source | `src/media/source_loop.cpp:90` | [14](14-sources-demux.md) |
| entrées clavier et souris appliquées | `src/app/play/chain.cpp:246` | [08](08-boucle-chaine.md) |
| réglage en direct | `src/app/play/chain.cpp:110` | [08](08-boucle-chaine.md) |
| réglage qui rouvre la chaîne | `src/app/play/chain.cpp:204` | [08](08-boucle-chaine.md) |
| changement de piste audio | `src/app/play/chain.cpp:69` | [08](08-boucle-chaine.md) |
| changement de sous-titres | `src/app/play/chain.cpp:88` | [08](08-boucle-chaine.md) |

### Fin d'épisode et reprise

| Point | Où | Zone |
|---|---|---|
| épisode voisin | `src/app/play/episodes.cpp:102` | [13](13-session-lecture.md) |
| passage au suivant | `src/app/play/player_run.cpp:720` | [08](08-boucle-chaine.md) |
| position retenue | `src/app/play/player_run.cpp:370` | [08](08-boucle-chaine.md) |
| position reprise | `src/app/play/resume.cpp:89` | [13](13-session-lecture.md) |

### Écran, veille et fermeture

| Point | Où | Zone |
|---|---|---|
| changement d'écran ou d'état HDR | `src/app/play/display_contract.cpp:16` | [11](11-ecran-materiel.md) |
| arrêt du journal | `src/core/log.cpp:164` | [20](20-socle-core.md) |

### Mise à jour

| Point | Où | Zone |
|---|---|---|
| vérification | `src/app/play/updater.cpp:384` | [26](26-mise-a-jour.md) |
| remplacement des fichiers | `src/app/play/updater.cpp:150` | [26](26-mise-a-jour.md) |
| retour arrière | `src/app/play/updater.cpp:217` | [26](26-mise-a-jour.md) |

> Aucun traitement de la mise en veille n'a été trouvé (ni `WM_POWERBROADCAST`, ni `PrepareForSleep`) : vérifier ce que devient une lecture après une veille.

## Défauts déjà connus de ce type

Relevés par le [contre-audit du 08/11](https://claude.ai/artifact/4rfGFb3P7zNiyHiwkfcnUU). Les revérifier, et chercher les autres.

| ID | Gravité | Défaut |
|---|---|---|
| GPU-03 | Majeur | Passer la fenêtre sur un écran d'une autre hauteur rouvre toute la chaîne, sans hystérésis. Sous TensorRT, cela peut lancer une construction de plusieurs minutes. |
| MED-01 | Majeur | Une erreur d'E/S de plus de 1,55 s devient une fin de flux, et l'épisode suivant démarre. La notice affiche « Playback goes on after an error ». |
| TRT-01 | Majeur | L'Auto calibré passe d'Économe à Léger sans vérifier que SPAN tient. Une source 1080p est réduite à 480p ou 540p sans agrandissement, et RIFE tourne à une autre taille que celle mesurée. |
| TRT-02 | Majeur | « Annuler et regarder » joue la vidéo réduite à la hauteur de travail, sans SPAN ni RIFE. `--no-upscale` a le même défaut. |
| UX-01 | Majeur | « Restore line art » et « Card for playback » manquent dans `live_part` et `change_between`. Ces réglages ne s'appliquent ni ne s'enregistrent jamais. |
| UX-02 | Majeur | L'assistant relancé pendant une lecture écrit `play.toml`, puis la chaîne rouvre avec les anciens réglages et les réécrit. Langue, mises à jour et chaîne calibrée sont perdues. |
| UX-03 | Majeur | Une vidéo transmise par un second lancement est jetée pendant l'assistant, la calibration, un téléchargement ou un export. Rien ne le signale. |

## Zones traversées

Les fiches de ces zones donnent la liste complète de leurs fichiers et leurs défauts connus. Les lire pour le contexte, sans refaire leur audit : un défaut local trouvé en chemin va dans « Renvois ».

- [06. Installation des moteurs et téléchargements](06-installation-moteurs.md)
- [08. Boucle du lecteur et ouverture de la chaîne](08-boucle-chaine.md)
- [11. Écran et matériel : EDID, VRR, HDR du système, télémétrie](11-ecran-materiel.md)
- [13. Session de lecture : seek, épisodes et reprise](13-session-lecture.md)
- [14. Sources, démultiplexage et erreurs d'E/S](14-sources-demux.md)
- [18. Démarrage et arrêt](18-demarrage-arret.md)
- [19. Intégration au système](19-integration-systeme.md)
- [20. Socle commun (core)](20-socle-core.md)
- [24. Interface : rendu, contrôles et messages](24-interface-rendu.md)
- [25. Assistant de premier lancement](25-assistant.md)
- [26. Mise à jour automatique et signature](26-mise-a-jour.md)
