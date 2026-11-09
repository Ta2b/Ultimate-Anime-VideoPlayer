# Complément de l'audit 39. Diagnostic et support

Branche : `claude/project-thread-9t0ajj`. Commit audité : `23ba02c`. Date : `2026-11-09`.

## Verdict

Sur les deux renvois en attente, l'un est un doublon d'A20-03 (nettoyage du journal limité au dossier
personnel). L'autre est confirmé et donne A39-10 : un disque plein pendant la construction d'un moteur
TensorRT est mesuré, mais l'utilisateur ne lit que « erreur BuilderFailed (détails dans le journal) ».

## Cadrage retenu

Aucune question à Tatoub, à la demande du coordinateur. Seuls les deux renvois de
`complements/39.md` sont vérifiés dans le code, sans GPU ni dépendances construites : aucun test lancé. Aucun
des deux renvois ne touche à la qualité d'image : il n'y a pas de baisse de qualité à signaler ici.

## Renvois traités

| De | Où | Constat du renvoi | Conclusion |
|---|---|---|---|
| 38 | `src/core/log_redact.cpp:52-73` | Nettoyage du journal à étendre aux racines de montage | **Doublon d'A20-03** (preuve ci-dessous) |
| 28 (complément) | `src/prepare/builder.cpp:233` | Disque plein soupçonné au journal seulement, pas nommé à l'utilisateur | **Confirmé** : A39-10 |

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A39-10 | Mineur | `src/prepare/builder.cpp:57-65`, `:233`, `src/app/play/visible_error.h:34-35`, `:44-45`, `src/app/play/player.cpp:596-598`, `src/app/main.cpp:537-542` | **Un disque plein pendant la construction d'un moteur n'est pas nommé à l'utilisateur.** Le code a tout ce qu'il faut : `judge` mesure la place libre du cache (`free_space`, `builder.cpp:33-37`) et écrit « Vérifiez la place » dans le refus, et `builder.cpp:233` soupçonne un disque plein quand le journal de trtexec ne s'écrit plus. Mais le refus garde le code `BuilderFailed`, et l'avertissement de la ligne 233 ne va qu'à `ajp.log`. Or `visible_error` ne montre que le code : `BuilderFailed` tombe dans le cas par défaut, « error BuilderFailed (details in the log) ». Le code `InsufficientSpace` existe, avec sa phrase traduite « not enough disk space », mais seul l'installateur des paquets s'en sert (`engine_install.cpp:229-233`). Le message de la ligne 233 se déclenche aussi sans disque plein (droits, chemin) et n'est jamais relié au refus. | CachyOS, cache des moteurs sur un disque presque plein (le cache n'a pas de purge, A33-04). Au changement de préréglage, trtexec échoue à écrire le moteur. Le lecteur essaie le cran Équilibre (`chain_setup.cpp:160-172`), qui échoue pareil. Ouvert depuis l'accueil, il y revient avec « "épisode.mkv" stopped: error BuilderFailed (details in the log) » ; ouvert par double-clic, une boîte dit la même chose. L'utilisateur ne sait pas qu'il doit libérer de la place, et le rapport qu'il envoie ne le dit qu'en octets bruts, au milieu de la ligne. | Dans `judge`, quand trtexec sort en erreur et que la place libre du dossier du moteur est sous un seuil (la taille du modèle ONNX fois un facteur, ou quelques centaines de Mo), ou quand le journal de trtexec n'a pas pu s'écrire (passer l'état de `log_file` à `judge`), rendre `RefusalCode::InsufficientSpace` avec la place libre et le dossier, en Mo. La phrase visible existe déjà. Ajouter au test de `builder` un trtexec simulé qui échoue sur un dossier sans place (le test manque déjà, A28-22). | Lu. `grep -rn InsufficientSpace src` : seuls `engine_install.cpp:233` et `visible_error.h:34`. |

## Défauts connus revérifiés

| ID | Statut | Note |
|---|---|---|
| A20-03 | Toujours présent | `log_redact.cpp:53-74` : un seul dossier (`main.cpp:408` ne passe que `place.home`), remplacé par sous-chaîne exacte. `/run/media/<user>/`, `/media/<user>/`, les noms 8.3 et la casse restent en clair. |
| A28-22 | Toujours présent | Aucun test de disque plein dans `tests/unit/prepare/`. A39-10 s'y ajoute. |
| A24-09 | Toujours présent | `visible_error.h:44-45` : `BuilderFailed` affiché par son nom anglais. A39-10 en est un cas, mais la correction (choisir le bon code) est propre au constructeur. |

## Preuves des doublons

- **Racines de montage (renvoi de 38).** `set_log_home` (`log_redact.cpp:53-67`) ne garde que les deux formes
  (`/` et `\`) d'un seul dossier, et `main.cpp:408` lui donne `place.home`. `redact_log_text`
  (`log_redact.cpp:69-74`) remplace ces formes par « ~ », à la casse près et sans frontière de composant. Rien
  ne vise `/run/media/<user>`, `/media/<user>`, ni les noms courts. C'est exactement A20-03 (« limité au
  dossier personnel. Le nom d'utilisateur reste dans `/run/media/<user>/` et `/media/<user>/` […], dans
  `C:\Users\JEAN-P~1\…` et dans `c:\users\…` »). Les chemins complets des vidéos, qui rendent ce manque visible
  dans le rapport, sont A39-03 et A38-01. La correction d'A20-03 doit ajouter `/run/media/<user>` et
  `/media/<user>` (le nom d'utilisateur tiré du dossier personnel) aux formes remplacées, sans attendre
  qu'A39-03 retire les chemins des vidéos : les moteurs, les modèles et `--data-dir` peuvent aussi y vivre.

## Renvois

Aucun nouveau renvoi.

## Points positifs

- Le refus d'une construction donne déjà le code de sortie, la dernière erreur de trtexec, le chemin du journal
  de construction et la place libre (`builder.cpp:57-65`) : le rapport a l'information, il ne lui manque que
  le bon code.
- Un moteur tronqué par un disque plein n'est jamais publié (voir le complément 28) : le défaut est
  d'ergonomie, pas d'intégrité.

## Questions ouvertes

Aucune.
