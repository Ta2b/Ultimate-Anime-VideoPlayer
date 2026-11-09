# Complément de l'audit 28. Renvois reçus des audits 27, 32, 33, 34, 35, 38 et 46

Branche : `claude/project-thread-3khvw0`. Commit audité : `d5df90d`. Date : `2026-11-09`.

## Verdict

Sur neuf renvois, quatre donnent un défaut nouveau (A28-18 à A28-21), dont un majeur : l'empreinte du paquet
TensorRT Linux ignore des entrées qui changent le module livré (architectures CUDA, préréglage Release, CUTLASS,
image de construction), si bien qu'un module périmé peut être réutilisé sans signal. Un cinquième (disque plein)
est réfuté dans le code mais n'a aucun test, ce qui donne A28-22. Les quatre autres sont des doublons de
`SYNTHESE.md`.

## Cadrage retenu

Pas de question posée à Tatoub, à la demande du coordinateur. Seuls les renvois de
`audits-docs/complements/28.md` sont vérifiés, en lisant le code au commit ci-dessus. Aucun code n'est modifié.

Qualité d'image : aucun de ces renvois ne contient de baisse de qualité nouvelle faite pour la vitesse. Le seul
lien avec la qualité, l'absence du SPAN FP8 sous Windows, est déjà relevé par A34-01.

## Conclusions par renvoi

| Renvoi | De | Conclusion |
|---|---|---|
| `tools/release/trt-fingerprint.sh:17-19` : empreinte sans `CMakeLists.txt` racine, presets ni CUTLASS | 27 | **Confirmé** : A28-18 |
| `.github/workflows/deps-watch.yml` : baseline vcpkg sans les versions Linux, libplacebo non bloqué | 27 | **Confirmé** pour la mécanique : A28-19. Le retard de libplacebo lui-même est un **doublon de A27-01** |
| `.github/workflows/release-linux.yml:67-70` : `restore-keys` qui restaure le préfixe d'une autre image | 27 | **Confirmé** : A28-20, même cause que A27-06 |
| `docs/auto-libre.md` § 3 : tableaux à dater et marquer historiques | 32 | **Confirmé** : A28-21, voisin de A32-01 |
| `.github/workflows/release-windows.yml:16-19` : PDB et CUTLASS absents de la release Windows | 34 | **Doublon** : PDB = A39-04, CUTLASS sous Windows = A34-01 |
| `src/prepare/cache.cpp` : disque plein, l'écriture d'un moteur doit refuser proprement | 33 | **Réfuté** dans le code (preuve ci-dessous). Cause (cache sans purge) = **doublon de A33-04**. Aucun test : A28-22 |
| `.github/dependabot.yml` : seules les actions GitHub suivies | 35 | **Doublon de A35-01** (veille des failles des dépendances) |
| `tools/release/notes/v0.10.3.md` : notes muettes sur l'absence de TensorRT Linux | 38 | **Doublon de A38-10**, point (2) |
| `THIRD_PARTY_NOTICES.md:49` : moteurs retirés encore listés | 46 | **Doublon de A38-09**, point (3) |

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A28-18 | Majeur | `tools/release/trt-fingerprint.sh:9-10,18-20`, `CMakeLists.txt:28-30,52-57`, `CMakePresets.json:200-212`, `src/gpu/cuda/backend/CMakeLists.txt:26-37`, `tools/release/preflight/Dockerfile:9,23` | L'empreinte qui décide si le paquet `trt-<empreinte>` doit être reconstruit ne hache ni le `CMakeLists.txt` racine (il fixe `AJP_CUDA_ARCHITECTURES`, la visibilité et `-fno-semantic-interposition`), ni `CMakePresets.json` (`linux-release` : Release et IPO), ni `third_party/cutlass` (compilé dans le module par `span_fp8_net.cu`), ni le `Dockerfile` du préflight (Ubuntu 24.04, `gcc-13`). Le commentaire exclut le `CMakeLists.txt` racine parce que sa `VERSION` change à chaque release. | On ajoute `100-real` aux architectures, on met CUTLASS à jour ou on change l'image : l'empreinte ne bouge pas, `trt-package.sh:21-23` répond « existe déjà : rien à faire », et Release Linux joint l'ancien module à la nouvelle version. Le module livré ne correspond plus au code, sans aucun message. | Ajouter aux entrées `CMakePresets.json`, `third_party/cutlass`, `tools/release/preflight/Dockerfile` et le `CMakeLists.txt` racine privé de sa ligne `VERSION` (par exemple `sed '/^  VERSION /d'` avant le hachage), puis corriger le commentaire. | Lu. |
| A28-19 | Mineur | `tools/ci/deps_lag.py:20-21,79-81`, `.github/workflows/deps-watch.yml:37-51` | La veille hebdomadaire ne fait de pull request que pour la baseline de `vcpkg.json`, donc pour Windows seul : les références Linux de `linux-from-source.sh` restent où elles sont. Quand cette pull request s'ouvre, la tâche finit en vert, même si d'autres dépendances (libplacebo aujourd'hui, 7.360.1 contre 7.387.0) restent en retard dans son seul tableau. Les ports en surcouche (`ports/libplacebo`, `libass`, `libunibreak`) ne suivent pas la baseline. Pourtant `libunibreak` est comparé au port de vcpkg, qui n'est pas celui construit. | Le bot fusionné fait passer FFmpeg ou SDL à une nouvelle version sous Windows et pas sous Linux : les deux paquets divergent sans test commun (A27-04). Le retard de libplacebo disparaît d'un tableau de pull request que personne ne relit. | Faire bouger ensemble la baseline et les entrées `DEPS` de même nom, ou refuser une baseline qui s'écarte de `DEPS`. Faire échouer la tâche tant qu'une ligne reste hors de la pull request. Retirer les ports en surcouche de `PORTS` et comparer leur `REF` à l'amont. | Lu. Versions de la baseline relevées : ffmpeg 9.0.2, sdl3 3.4.20, imgui 1.93.0, identiques à `DEPS` aujourd'hui. |
| A28-20 | Mineur | `.github/workflows/release-linux.yml:64-70` (et `ci.yml:69-70,121-122,240-241`) | La clé du cache des dépendances de la release ne porte que `gcc-13` et le hachage de `linux-from-source.sh`. `restore-keys: deps-player-gcc-13-` restaure donc n'importe quel préfixe plus ancien, construit sur une autre image `ubuntu-24.04` (en-têtes du système, `gcc-13` mis à jour, paquets de `versions.env`). Les marqueurs ne rebâtissent que sur un changement de référence ou d'option (A27-06). | Après une mise à jour de l'image du runner, la release lie un FFmpeg ou une SDL compilés contre les en-têtes de l'image précédente. Un changement de `APT_PLAYER` dans `versions.env` ne rebâtit rien non plus. | Pour la release, ajouter `${{ env.ImageOS }}-${{ env.ImageVersion }}` et `hashFiles('tools/release/versions.env')` à la clé et à la seule `restore-keys` gardée, ou reconstruire le préfixe à neuf pour une étiquette. | Lu. |
| A28-21 | Mineur | `docs/auto-libre.md:140-174`, `docs/README.md:14` | La carte des documents range `auto-libre.md` parmi ceux qui « décrivent le code actuel ». Pourtant, ses tableaux du § 3 n'ont pas de date ni de commit. Ils comparent un GMFSS retiré des sources (ligne 170). Ils renvoient à un banc dans `/tmp/claude-1000/pbt/cmp3.py`, hors du dépôt, et à `origin/auto-mesure` (#123). | Un lecteur, humain ou IA, prend 60 i/s et 13 à 16 ms de service pour l'état actuel et calibre une décision dessus (c'est déjà arrivé : A32-01). | Dater le § 3 (date et commit), le titrer « historique », retirer ou marquer la ligne GMFSS, verser le banc dans `tools/bench/` ou citer qu'il est perdu. | Lu. |
| A28-22 | Mineur | `tests/unit/prepare/` (aucun test), `src/prepare/cache.cpp:117-133` | Aucun test ne fait échouer l'écriture d'un moteur faute de place. Le bon comportement, lu dans le code (voir plus bas), n'est donc protégé par rien en CI. | Un changement de `produce_once` ou de `publish_file` qui publierait un moteur tronqué passerait la CI. | Ajouter au test de `cache.cpp` un producteur qui écrit un fichier tronqué puis renvoie l'erreur `CacheIo`, et un `EngineCheck` qui refuse. Vérifier qu'aucun fichier n'est publié et que le dossier de travail est supprimé. Sous Linux, un test optionnel sur un petit `tmpfs`. | Lu (`grep -i "plein\|ENOSPC" tests` : rien). |

## Preuves des réfutations et des doublons

- **Disque plein (renvoi de 33).** Le moteur s'écrit dans un dossier de travail (`cache.cpp:117-120`), supprimé
  à la sortie quel que soit le chemin (`StageGuard`, `cache.cpp:23-30`). Avant la publication, un moteur
  tronqué est refusé de trois façons. Un fichier vide est refusé (`builder.cpp:238-242`, `cache.cpp:124-127`).
  Le moteur doit aussi se désérialiser (`EngineCheck`, `builder.cpp:250-252`, posé par `tensorrt_backend.cpp:324-330`,
  `tensorrt_rife_fused.cpp:35` et `tensorrt_drba_fused.cpp:35`). Enfin, `fsync` est vérifié avant le renommage
  (`cache_io.cpp:92-96`). L'échec de l'empreinte `.sha256` est ignoré sans risque : la lecture suivante la
  refait (`cache.cpp:40-57`). Un timing cache non gardé n'est qu'un avertissement (`builder.cpp:95-101`).
  Reste un défaut d'ergonomie, pas d'intégrité : l'utilisateur voit l'échec de trtexec ou de la désérialisation,
  pas « disque plein ». Le journal le soupçonne seulement (`builder.cpp:233`). La cause, un cache sans purge,
  est A33-04.
- **PDB et CUTLASS sous Windows (renvoi de 34).** `windows-release` est en `Release` sans `/Zi` (`CMakePresets.json:312`),
  aucun workflow ne produit ni ne garde de PDB, et `crash_dump.cpp:47-50` écrit des minidumps : c'est A39-04.
  `span_fp8` est faux par défaut (`release-windows.yml:16-19`) et `AJP_WITH_SPAN_FP8` OFF hors Linux
  (`src/gpu/cuda/backend/CMakeLists.txt:27-31`) : c'est A34-01. Un point s'ajoute au dossier : une étiquette
  poussée n'a pas d'`inputs`, donc ni TensorRT ni FP8 sous Windows (`release-windows.yml:74,99,168`). Le
  constat est le même que celui de A28-17, sans nouvel ID.
- **Dependabot (renvoi de 35).** `.github/dependabot.yml:5` ne suit que `github-actions`. Le dépôt n'a ni
  `requirements.txt`, ni `pyproject.toml`, ni `package.json`. Les dépendances natives (`linux-from-source.sh`,
  ports en surcouche) échappent à Dependabot. Leur veille des failles est A35-01, et `deps-watch.yml` ne suit
  que le retard de version (A28-19).
- **Notes 0.10.3 (renvoi de 38).** `tools/release/notes/v0.10.3.md:1` annonce Linux et les moteurs « téléchargés
  à part », sans limite connue pour TensorRT Linux. C'est le point (2) de A38-10, dont la cause est A28-01.
- **Notices (renvoi de 46).** `THIRD_PARTY_NOTICES.md:48-49` liste toujours OpenVINO, MIGraphX, ROCm, ONNX
  Runtime et Windows ML. C'est le point (3) de A38-09.

## Renvois

| Vers l'audit | Où | Constat |
|---|---|---|
| 39 | `src/prepare/builder.cpp:233` | Un disque plein n'est soupçonné qu'au journal : le refus montré à l'utilisateur ne le nomme pas |
