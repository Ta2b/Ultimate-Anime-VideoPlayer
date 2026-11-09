# Audit 18. Démarrage et arrêt

**Audit par zone. Priorité : haute.** 18 fichiers, 3 327 lignes ; tests rattachés : 6 fichiers, 840 lignes.

## Mission

Du lancement à la première image et de la dernière image à la sortie : options, ordre d'initialisation, accueil, attente, arrêt de tous les composants.

## Points à auditer en profondeur

Un point de départ : l'audit doit aussi chercher ce que la liste ne nomme pas.

- Ordre de démarrage et échec de chaque étape (réglages illisibles, DLL manquante, Vulkan absent).
- Options de ligne de commande : chaque option fait ce que dit l'aide (TRT-02).
- Arrêt : aucun thread bloqué, aucun journal après `spdlog::shutdown` (QUA-13).
- Ordre de recherche des DLL (SEC-25).
- Vérification des mises à jour lancée au démarrage : thread, délai.

## Frontières

Dépend de :

- [20. Socle commun (core)](20-socle-core.md) (19 inclusions)
- [24. Interface : rendu, contrôles et messages](24-interface-rendu.md) (8 inclusions)
- [08. Boucle du lecteur et ouverture de la chaîne](08-boucle-chaine.md) (7 inclusions)
- [07. Calibration, niveaux et mode Auto](07-calibration-niveaux.md) (6 inclusions)
- [19. Intégration au système](19-integration-systeme.md) (5 inclusions)
- [14. Sources, démultiplexage et erreurs d'E/S](14-sources-demux.md) (5 inclusions)
- [17. Sous-titres](17-sous-titres.md) (5 inclusions)
- [12. Cadence et horloges](12-cadence-horloges.md) (5 inclusions)
- [13. Session de lecture : seek, épisodes et reprise](13-session-lecture.md) (4 inclusions)
- [22. Réglages : modèle, application et persistance](22-reglages-modele.md) (3 inclusions)
- [26. Mise à jour automatique et signature](26-mise-a-jour.md) (3 inclusions)
- [16. Audio](16-audio.md) (3 inclusions)
- [10. Rendu et présentation par libplacebo](10-presentation-libplacebo.md) (3 inclusions)
- [06. Installation des moteurs et téléchargements](06-installation-moteurs.md) (3 inclusions)
- [04. Analyse et découpe des modèles ONNX](04-onnx-modeles.md) (2 inclusions)
- [43. Textes et traductions](43-textes-traductions.md) (2 inclusions)
- [11. Écran et matériel : EDID, VRR, HDR du système, télémétrie](11-ecran-materiel.md) (2 inclusions)
- [25. Assistant de premier lancement](25-assistant.md) (2 inclusions)
- [09. Étages d'agrandissement et d'interpolation](09-etages-traitement.md) (2 inclusions)
- [21. Frontière des moteurs (interface et ABI)](21-frontiere-moteurs.md) (1 inclusion)
- [47. Export vidéo](47-export.md) (1 inclusion)
- [41. HDR, grade et looks](41-hdr-grade.md) (1 inclusion)
- [05. Construction des moteurs, cache et trtexec](05-construction-moteurs.md) (1 inclusion)

Utilisé par :

- [08. Boucle du lecteur et ouverture de la chaîne](08-boucle-chaine.md) (8 inclusions)
- [24. Interface : rendu, contrôles et messages](24-interface-rendu.md) (6 inclusions)
- [25. Assistant de premier lancement](25-assistant.md) (4 inclusions)
- [13. Session de lecture : seek, épisodes et reprise](13-session-lecture.md) (4 inclusions)
- [47. Export vidéo](47-export.md) (3 inclusions)
- [22. Réglages : modèle, application et persistance](22-reglages-modele.md) (3 inclusions)
- [11. Écran et matériel : EDID, VRR, HDR du système, télémétrie](11-ecran-materiel.md) (3 inclusions)
- [09. Étages d'agrandissement et d'interpolation](09-etages-traitement.md) (2 inclusions)
- [10. Rendu et présentation par libplacebo](10-presentation-libplacebo.md) (2 inclusions)
- [07. Calibration, niveaux et mode Auto](07-calibration-niveaux.md) (1 inclusion)
- [06. Installation des moteurs et téléchargements](06-installation-moteurs.md) (1 inclusion)
- [41. HDR, grade et looks](41-hdr-grade.md) (1 inclusion)
- [23. Réglages : le panneau](23-reglages-panneau.md) (1 inclusion)
- [28. CI, publication et signature](28-ci-publication.md) (1 inclusion)

Auditer l'interface avec ces zones (ce qui entre, ce qui sort, qui possède quoi, quel thread appelle), pas leur intérieur. Les parcours complets sont couverts par les audits transverses.

## Défauts déjà connus

Relevés par le [contre-audit du 08/11](https://claude.ai/artifact/4rfGFb3P7zNiyHiwkfcnUU). Les revérifier un par un, sans s'y limiter.

| ID | Gravité | Défaut |
|---|---|---|
| GPU-08 | Majeur | La compression P0.16 ne joue pas sur macOS. Sous Wayland hors KDE, elle suppose un blanc à 203 nits. |
| QUA-04 | Majeur | Les résultats ignorés ne sont vérifiés ni sous Windows, ni hors des PR. Les sites relevés le 06/11 sont intacts. |
| QUA-07 | Majeur | Sous Windows, l'échec d'écriture de `update.toml` (antivirus) est ignoré. Le lancement compte comme un plantage, et la version peut être retirée après 3 lancements. |
| QUA-15 | Majeur | Aucune fonction de CCN > 50 n'a été découpée, et le code neuf y a été ajouté (`play` passe de 105 à 133). On compte 23 fonctions de CCN > 50 contre 22. |
| SEC-06 | Majeur | La garde confirme une version dès une sortie propre, même sans lecture. Une sortie sur erreur fatale ou une mort avant la garde ne compte jamais : une release cassée reste installée. |
| TRT-02 | Majeur | « Annuler et regarder » joue la vidéo réduite à la hauteur de travail, sans SPAN ni RIFE. `--no-upscale` a le même défaut. |
| UX-02 | Majeur | L'assistant relancé pendant une lecture écrit `play.toml`, puis la chaîne rouvre avec les anciens réglages et les réécrit. Langue, mises à jour et chaîne calibrée sont perdues. |
| WIN-02 | Majeur | Avec un profil redirigé, la migration copie plus de 1 Go sur le thread principal, avant la fenêtre. Le lecteur semble ne pas démarrer pendant des minutes. |
| GPU-15 | Mineur | L'anneau RGB10A2 est le défaut sous TensorRT, avant un grade qui étire le contraste. Des bandes sont possibles dans les dégradés. |
| MED-08 | Mineur | Au retrait d'un appareil, son délai reste appliqué aux haut-parleurs. Sans sortie par défaut, chaque bloc échoue au journal et rien n'est retenté. |
| QUA-11 | Mineur | `main.cpp` écrit encore `data / "update.toml"` cinq fois malgré `paths::update_state`. D'autres noms restent en dur. |
| QUA-13 | Mineur | La sonde abandonnée reste un thread détaché. Elle peut journaliser après la destruction des statiques. |
| SEC-09 | Mineur | Trois fils lisent et réécrivent `update.toml` sans verrou. Un téléchargement ou une révocation peut être perdu. |
| SEC-25 | Mineur | L'ordre de recherche des DLL n'est pas restreint, et les mises à jour n'ont pas de Mark-of-the-Web. |
| TRT-18 | Mineur | Dossiers temporaires à nom fixe, partagés entre utilisateurs puis effacés par `remove_all`. |
| UX-17 | Mineur | Le message d'erreur se répète (« ne se lit pas : le fichier ne se lit pas »). La boîte fatale renvoie deux fois au journal. |
| WIN-03 | Mineur | Jusqu'à environ 192 Mo de journal et 5 minidumps restent dans le profil itinérant. |
| WIN-11 | Mineur | Écarts inchangés : pré-analyse sur un thread, sémaphores non partagés, `timeBeginPeriod(1)` global, manifeste sans `<supportedOS>`, veille jamais suspendue, télémétrie NVIDIA seule. |

## Fichiers

| Fichier | Lignes |
|---|---|
| `src/app/CMakeLists.txt` | 93 |
| `src/app/main.cpp` | 556 |
| `src/app/play/bridges.cpp` | 136 |
| `src/app/play/bridges.h` | 100 |
| `src/app/play/links.h` | 174 |
| `src/app/play/loop_wait.cpp` | 60 |
| `src/app/play/loop_wait.h` | 27 |
| `src/app/play/mapped_file.cpp` | 51 |
| `src/app/play/mapped_file.h` | 28 |
| `src/app/play/options.cpp` | 639 |
| `src/app/play/options.h` | 274 |
| `src/app/play/paths.h` | 65 |
| `src/app/play/player.cpp` | 646 |
| `src/app/play/player.h` | 57 |
| `src/app/play/pumped_task.cpp` | 20 |
| `src/app/play/pumped_task.h` | 77 |
| `src/app/play/wiring.cpp` | 183 |
| `src/app/play/wiring.h` | 141 |

## Tests rattachés

- `tests/unit/app/loop_wait_test.cpp`
- `tests/unit/app/options_test.cpp`
- `tests/unit/app/paths_test.cpp`
- `tests/unit/app/pumped_task_test.cpp`
- `tests/unit/app/wiring_test.cpp`
- `tests/unit/audio/player_test.cpp`

## Hors de cet audit

Tout fichier non listé ci-dessus. Un défaut trouvé ailleurs va dans « Renvois » du rapport, avec le numéro de l'audit concerné.
