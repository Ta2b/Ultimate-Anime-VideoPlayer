# Audit 19. Intégration au système

**Audit par zone. Priorité : haute.** 10 fichiers, 1 417 lignes ; tests rattachés : 4 fichiers, 375 lignes.

## Mission

Ce qui relie le lecteur au système : instance unique, minidumps, inscription comme lecteur par défaut, dossiers de données et leur migration, langue du système.

## Points à auditer en profondeur

Un point de départ : l'audit doit aussi chercher ce que la liste ne nomme pas.

- Instance unique : canal nommé Windows, course sous Linux, fichier perdu pendant l'assistant (SEC-15, WIN-06, UX-03).
- Migration des données vers `%LOCALAPPDATA%` : thread, progression, coupure (WIN-02, WIN-03).
- Purge des dossiers « mal placés » : fichiers d'autres programmes (SEC-16).
- Minidumps : symboles, `__fastfail`, chemins (WIN-07).
- Lecteur par défaut : état réel sous Windows et sous CachyOS, sans sous-processus bloquant.
- Langue du système détectée pour la langue de l'interface.

## Frontières

Dépend de :

- [20. Socle commun (core)](20-socle-core.md) (5 inclusions)
- [14. Sources, démultiplexage et erreurs d'E/S](14-sources-demux.md) (3 inclusions)
- [04. Analyse et découpe des modèles ONNX](04-onnx-modeles.md) (2 inclusions)
- [43. Textes et traductions](43-textes-traductions.md) (1 inclusion)
- [05. Construction des moteurs, cache et trtexec](05-construction-moteurs.md) (1 inclusion)

Utilisé par :

- [18. Démarrage et arrêt](18-demarrage-arret.md) (5 inclusions)
- [24. Interface : rendu, contrôles et messages](24-interface-rendu.md) (2 inclusions)
- [22. Réglages : modèle, application et persistance](22-reglages-modele.md) (2 inclusions)
- [23. Réglages : le panneau](23-reglages-panneau.md) (2 inclusions)
- [25. Assistant de premier lancement](25-assistant.md) (1 inclusion)
- [13. Session de lecture : seek, épisodes et reprise](13-session-lecture.md) (1 inclusion)
- [08. Boucle du lecteur et ouverture de la chaîne](08-boucle-chaine.md) (1 inclusion)

Auditer l'interface avec ces zones (ce qui entre, ce qui sort, qui possède quoi, quel thread appelle), pas leur intérieur. Les parcours complets sont couverts par les audits transverses.

## Défauts déjà connus

Relevés par le [contre-audit du 08/11](https://claude.ai/artifact/4rfGFb3P7zNiyHiwkfcnUU). Les revérifier un par un, sans s'y limiter.

| ID | Gravité | Défaut |
|---|---|---|
| UX-03 | Majeur | Une vidéo transmise par un second lancement est jetée pendant l'assistant, la calibration, un téléchargement ou un export. Rien ne le signale. |
| WIN-02 | Majeur | Avec un profil redirigé, la migration copie plus de 1 Go sur le thread principal, avant la fenêtre. Le lecteur semble ne pas démarrer pendant des minutes. |
| WIN-07 | Majeur | Les minidumps sont sans PDB et ne voient ni `__fastfail` ni débordement de pile. Ils gardent les chemins en clair, et Linux n'a rien. |
| SEC-15 | Mineur | Le canal nommé Windows a un nom prévisible et l'ACL par défaut. Sur un poste partagé, un autre utilisateur capte les vidéos ouvertes ou bloque le lecteur (voir WIN-06). |
| SEC-16 | Mineur | `purge_misplaced` efface à chaque lancement les dossiers qui « ressemblent à des versions » dans `%APPDATA%` ou `~/.local/share`. D'autres programmes peuvent y perdre des données. |
| WIN-06 | Mineur | Sous Windows, un client muet bloque la fermeture. Sous Linux, une course entre `bind` et `listen` peut ouvrir deux fenêtres (voir SEC-15). |

## Fichiers

| Fichier | Lignes |
|---|---|
| `src/app/play/crash_dump.cpp` | 102 |
| `src/app/play/crash_dump.h` | 17 |
| `src/app/play/data_dirs.cpp` | 141 |
| `src/app/play/data_dirs.h` | 29 |
| `src/app/play/default_player.cpp` | 412 |
| `src/app/play/default_player.h` | 92 |
| `src/app/play/single_instance.cpp` | 418 |
| `src/app/play/single_instance.h` | 48 |
| `src/app/play/system_language.cpp` | 123 |
| `src/app/play/system_language.h` | 35 |

## Tests rattachés

- `tests/unit/app/crash_dump_test.cpp`
- `tests/unit/app/default_player_test.cpp`
- `tests/unit/app/single_instance_test.cpp`
- `tests/unit/app/system_language_test.cpp`

## Hors de cet audit

Tout fichier non listé ci-dessus. Un défaut trouvé ailleurs va dans « Renvois » du rapport, avec le numéro de l'audit concerné.
