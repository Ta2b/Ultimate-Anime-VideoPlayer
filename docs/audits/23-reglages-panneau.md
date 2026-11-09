# Audit 23. Réglages : le panneau

**Audit par zone. Priorité : haute.** 12 fichiers, 2 324 lignes ; tests rattachés : 0 fichiers, 0 lignes.

## Mission

Les pages du panneau de réglages : image, mouvement, son, moteurs, système, et leur ordre.

## Points à auditer en profondeur

Un point de départ : l'audit doit aussi chercher ce que la liste ne nomme pas.

- Ordre et regroupement : vue de base et avancée.
- Réglages en direct contre réglages qui rouvrent la vidéo : signalés ?
- Accès disque et processus à chaque image (pages Application et Export).
- Réinitialisation : confirmation, langue gardée.
- Dossier des moteurs coupé à 511 octets (QUA-14).
- Textes : notes et infobulles.

## Frontières

Dépend de :

- [24. Interface : rendu, contrôles et messages](24-interface-rendu.md) (13 inclusions)
- [07. Calibration, niveaux et mode Auto](07-calibration-niveaux.md) (6 inclusions)
- [22. Réglages : modèle, application et persistance](22-reglages-modele.md) (5 inclusions)
- [09. Étages d'agrandissement et d'interpolation](09-etages-traitement.md) (4 inclusions)
- [06. Installation des moteurs et téléchargements](06-installation-moteurs.md) (3 inclusions)
- [14. Sources, démultiplexage et erreurs d'E/S](14-sources-demux.md) (3 inclusions)
- [47. Export vidéo](47-export.md) (2 inclusions)
- [43. Textes et traductions](43-textes-traductions.md) (2 inclusions)
- [41. HDR, grade et looks](41-hdr-grade.md) (2 inclusions)
- [19. Intégration au système](19-integration-systeme.md) (2 inclusions)
- [20. Socle commun (core)](20-socle-core.md) (2 inclusions)
- [25. Assistant de premier lancement](25-assistant.md) (1 inclusion)
- [04. Analyse et découpe des modèles ONNX](04-onnx-modeles.md) (1 inclusion)
- [18. Démarrage et arrêt](18-demarrage-arret.md) (1 inclusion)
- [26. Mise à jour automatique et signature](26-mise-a-jour.md) (1 inclusion)

Utilisé par :

- [08. Boucle du lecteur et ouverture de la chaîne](08-boucle-chaine.md) (1 inclusion)
- [25. Assistant de premier lancement](25-assistant.md) (1 inclusion)
- [22. Réglages : modèle, application et persistance](22-reglages-modele.md) (1 inclusion)
- [07. Calibration, niveaux et mode Auto](07-calibration-niveaux.md) (1 inclusion)
- [09. Étages d'agrandissement et d'interpolation](09-etages-traitement.md) (1 inclusion)
- [24. Interface : rendu, contrôles et messages](24-interface-rendu.md) (1 inclusion)

Auditer l'interface avec ces zones (ce qui entre, ce qui sort, qui possède quoi, quel thread appelle), pas leur intérieur. Les parcours complets sont couverts par les audits transverses.

## Défauts déjà connus

Relevés par le [contre-audit du 08/11](https://claude.ai/artifact/4rfGFb3P7zNiyHiwkfcnUU). Les revérifier un par un, sans s'y limiter.

| ID | Gravité | Défaut |
|---|---|---|
| TRT-07 | Majeur | Le menu propose ArtCNN en FP8, toujours refusé. Le fichier « fp16 » calcule en FP32, sous TensorRT comme sous Vulkan (rapport G). |
| UX-13 | Majeur | « Report a problem » promet un journal sans dossiers, mais les chemins complets hors du dossier personnel y restent (voir WIN-08). |
| QUA-14 | Mineur | Le dossier des moteurs est coupé à 511 octets, au milieu d'un caractère UTF-8. |
| SEC-08 | Mineur | « Install now » applique sans revérifier la version. Il peut réinstaller la 0.10.1 par-dessus une 0.10.2. |
| SEC-09 | Mineur | Trois fils lisent et réécrivent `update.toml` sans verrou. Un téléchargement ou une révocation peut être perdu. |
| UX-10 | Mineur | Le point décimal devient une virgule sans condition de langue. L'anglais affiche « 23,976 fps ». |
| UX-16 | Mineur | La note « Greyed out… » s'affiche en lecture alors que rien n'est grisé. |
| UX-18 | Mineur | Le curseur du délai des sous-titres va à ±5 s, les touches z et x jusqu'à ±30 s. |
| UX-19 | Mineur | Le dossier des moteurs est enregistré à chaque frappe. Quitter pendant la saisie garde un chemin à moitié tapé. |
| UX-21 | Mineur | `bench_page` (environ 120 lignes) n'a aucun appelant, et 186 entrées de traduction sont orphelines. |

## Fichiers

| Fichier | Lignes |
|---|---|
| `src/app/play/settings_chain_tabs.cpp` | 402 |
| `src/app/play/settings_chain_tabs.h` | 16 |
| `src/app/play/settings_engines.cpp` | 94 |
| `src/app/play/settings_engines.h` | 22 |
| `src/app/play/settings_export.cpp` | 225 |
| `src/app/play/settings_export.h` | 30 |
| `src/app/play/settings_panel.cpp` | 603 |
| `src/app/play/settings_panel.h` | 189 |
| `src/app/play/settings_sound.cpp` | 91 |
| `src/app/play/settings_sound.h` | 15 |
| `src/app/play/settings_system.cpp` | 513 |
| `src/app/play/settings_system.h` | 124 |

## Hors de cet audit

Tout fichier non listé ci-dessus. Un défaut trouvé ailleurs va dans « Renvois » du rapport, avec le numéro de l'audit concerné.
