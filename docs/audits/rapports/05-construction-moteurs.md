# Rapport de l'audit 05. Construction des moteurs, cache et trtexec

Branche : `claude/project-thread-ysanx6` (la branche du fil ; `audit/05-construction-moteurs` n'était pas
ouverte en écriture). Commit audité : `7739971`. Date : `2026-11-09`.

## Verdict

Le cœur est solide (publication atomique, verrous qui ne se perdent pas, désérialisation avant publication,
lanceur sans shell), mais sous CachyOS un lecteur qui meurt pendant une construction laisse `trtexec` tourner
jusqu'à 20 minutes, ce qui fausse le choix des tactiques de la construction suivante (A05-01, démontré). Le
reste est mineur : délais incohérents entre eux, aucune purge du cache, aucun repli si le timing cache est
refusé, et environ 700 lignes de code mort.

## Cadrage retenu

Deux questions posées à Tatoub en début de fil. Le fil a suivi les options recommandées sans attendre ;
Tatoub les a confirmées toutes les deux :

- `preparer` et `receipt` : seulement vérifiés comme morts et relevés pour retrait (QUA-03), sans audit détaillé
  (choix de Tatoub) ;
- les deux usages couverts à égalité : premier lancement et changement de pilote (clé, reconstructions, délai,
  reprise après échec) d'un côté, tailles de vidéo variées de l'autre (choix de Tatoub).

Le fil tourne sans carte NVIDIA et sans les dépendances du dépôt : le lanceur POSIX a été compilé seul et mis à
l'épreuve par un programme autonome ; le reste est lu, ou « à confirmer » avec la manipulation.

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A05-01 | Majeur | `src/prepare/process_posix.cpp:127` | Sous Linux, rien ne tue `trtexec` si le lecteur meurt. Il est mis dans son propre groupe (Ctrl+C du terminal ne l'atteint pas) et seul `run_process` le tue (`:198`), à l'annulation ou au délai. Windows a l'objet Job `KILL_ON_JOB_CLOSE` (`process_win.cpp:101-107`) ; POSIX n'a aucun équivalent. | CachyOS, premier lancement : le lecteur plante, est tué (`kill -9`, « Forcer à quitter ») ou la session se ferme pendant la construction du SPAN. `trtexec` continue jusqu'à 20 min. Le `flock` du GPU (`builder.cpp:71-78`), lui, est rendu par le noyau : le lecteur relancé démarre une seconde construction en même temps que l'orpheline. C'est exactement ce que le verrou du GPU devait empêcher (`builder.h:53-60`) : les chronométrages de tactiques sont faussés, le moteur publié et le timing cache gardent des choix plus lents, pour toujours (la clé ne change pas). | Lier l'enfant à la vie du lecteur : sous Linux, `fork` puis `prctl(PR_SET_PDEATHSIG, SIGKILL)` avant `exec` (avec la vérification `getppid()` d'usage), ou un petit tremplin qui le fait ; à défaut, tuer le groupe dans le gestionnaire de plantage (`crash_dump.cpp`) et à la sortie. | Démontré : programme autonome compilé avec `process_posix.cpp` et `process_text.cpp`, un « lecteur » lance `sh -c 'exec sleep 30'` par `run_process` puis se tue par SIGKILL : l'enfant est « TOUJOURS VIVANT » une seconde après. Test à ajouter à `process_test.cpp` : même montage par `fork`, attendre que le petit-enfant meure dans les 2 s. |
| A05-02 | Mineur | `src/prepare/key_lock.h:26` | Les délais ne s'emboîtent pas. Le verrou de la clé est tenu pendant toute la production, attente du verrou du GPU comprise (`builder.cpp:203`, jusqu'à 45 min) puis `trtexec` (jusqu'à 20 min, `builder.h:35`) ; un second lecteur qui attend la même clé abandonne pourtant à 45 min (`kMaxLockWait`). | Deux lecteurs ouverts sur la même vidéo pendant qu'un troisième construit RIFE fusionné (six morceaux) : le premier attend le GPU, le deuxième attend la clé et reçoit au bout de 45 min « verrou tenu… fermez les autres lecteurs ajp », alors que rien n'est figé. | Donner au verrou de la clé un délai d'au moins l'attente du GPU plus le délai de `trtexec`, ou ne compter que le temps où le détenteur ne progresse pas (un battement écrit dans le fichier verrou). | Lu |
| A05-03 | Mineur | `src/prepare/builder.cpp:86-93` | Aucun repli si le timing cache persistant fait échouer `trtexec`. Il n'est remplacé qu'après une construction réussie (`:253`) : s'il est refusé, chaque construction suivante échoue sur ce GPU, pour tous les modèles et toutes les tailles. | Coupure de courant ou disque défaillant qui abîme `aji-timing-<id>.cache` malgré le `fsync` ; ou trtexec qui refuse un cache écrit par un build précédent de même version. Chaque ouverture d'une taille nouvelle finit en `BuilderFailed`, et le refus ne parle pas du timing cache. | Sur un échec `Exited` avec un timing cache copié, relancer une fois sans lui ; s'il réussit, remplacer le cache persistant. Journaliser le cas. | À confirmer : sur la machine de Tatoub, remplacer `aji-timing-*.cache` par 1 Ko d'octets aléatoires, ouvrir une vidéo d'une taille jamais vue, lire `*.build.log`. |
| A05-04 | Mineur | `src/prepare/cache_io.cpp:95` | La publication renomme une seule fois. Sous Windows, un antivirus ou l'indexeur qui ouvre le moteur juste écrit fait échouer `MoveFileEx` (accès refusé, violation de partage) : la construction entière est jetée. | Windows 11 avec Defender actif : `model.engine` (dizaines de Mo) vient d'être fermé par trtexec puis relu par `check_` ; Defender l'analyse, le renommage échoue, le refus « publication : … accès refusé » arrive après une minute de construction, et la suivante recommence. | Sous Windows, quelques nouveaux essais espacés (par exemple 5 × 200 ms) sur `ERROR_ACCESS_DENIED` et `ERROR_SHARING_VIOLATION` avant de refuser. Même chose pour le temporaire de l'empreinte (`core::write_text_atomically`). | À confirmer : Windows 11, Defender en temps réel, vider le cache des moteurs, ouvrir dix vidéos de tailles différentes et chercher « publication » dans le journal. |
| A05-05 | Mineur | `src/prepare/trtexec_command.cpp:162-169` | L'empreinte de `trtexec` n'est vérifiée qu'une fois, à la création du backend (`src/gpu/cuda/backend/entry.cpp:58`), pas juste avant le lancement ; et le repli de développement est rendu sans aucune vérification. Le repli n'est retiré qu'en `Release` (SEC-20), or `linux-prefix-cuda-release` et `-machine`, le lecteur de tous les jours, sont en `RelWithDebInfo`. | Le `trtexec` du paquet manque ou a été remplacé (mise à jour à moitié faite) : le lecteur de Tatoub prend en silence celui du poste, d'une autre version de TensorRT que la bibliothèque chargée. `check_` refuse alors le moteur après la construction : une minute perdue à chaque ouverture, avec un refus qui ne dit pas qu'un autre `trtexec` a servi. | Revérifier l'empreinte dans `TrtexecBuilder::run` juste avant `run_process` ; journaliser en avertissement l'usage du repli ; retirer le repli hors `Debug` (voir le renvoi vers 27). | Lu |
| A05-06 | Mineur | `src/prepare/cache.cpp:52-53` | `intact()` écrit l'empreinte d'une entrée qui n'en a pas sans tenir le verrou (chemin rapide, `:99-101`). Un processus qui reprend l'ancienne entrée peut écrire son empreinte après celle du remplaçant publié par un autre. | Une entrée d'avant l'empreinte (ou dont l'empreinte n'a pas pu s'écrire, `store_digest` ignore l'échec) est jugée abîmée par A, qui la reconstruit sous le verrou ; B la lisait au même moment sans verrou et écrit l'empreinte de l'ancienne. Au lancement suivant, l'empreinte ne correspond plus : nouvelle construction inutile. | N'écrire une empreinte manquante que sous le verrou de la clé (passer par le chemin lent quand elle manque). | Lu |
| A05-07 | Mineur | `src/prepare/trtexec_progress.cpp:21-22` | Aucun repère entre 10 % et 85 % : la barre reste sur « Choix des tactiques, 10 % » pendant l'essentiel de la construction (53 à 64 s sur 70 pour le SPAN, d'après le commentaire `:16-17`). | Premier lancement sur un portable : la barre semble figée plusieurs minutes, l'utilisateur ferme le lecteur… et retombe sur A05-01 sous Linux. | Interpoler sur le temps attendu (que `build_progress.cpp` sait déjà estimer depuis les journaux), ou compter les lignes de tactiques de trtexec en verbeux. | Lu ; à confirmer sur trtexec 11.3 (le commentaire `trtexec_progress.h:5` le demande). |
| A05-08 | Mineur | `src/prepare/preparer.cpp`, `src/prepare/receipt.cpp`, `src/prepare/receipt_fields.cpp` | Code mort compilé et testé : `preparer` (220 lignes), `receipt` et `receipt_fields` (474 lignes) n'ont aucun appelant hors des tests (QUA-03). Dans le code vivant aussi : `TrtexecBuilder::rebuild` et tout le chemin `rejected_sha256` (`builder.h:74-79`, `cache.h:30-34`, `cache.cpp:61-69`) ne sont appelés que par `build()` avec une valeur vide ; `has_owned_stages` (`cache_io.cpp:151`) n'a aucun appelant ; `write_atomic` (`cache_io.cpp:101`) double `core::write_text_atomically` ; `ProgramFormat::MiGraphX` et `OrtProvider` (`identity.h:25`) nomment des moteurs retirés. Le commentaire `src/app/play/engine_build.h:5` annonce encore que `preparer` remplacera `EngineBuild`. | Chaque lecteur du code croit qu'un moteur refusé au chargement est reconstruit (`builder.h:74`) : ce n'est pas le cas, aucun appelant ne passe l'empreinte refusée. | Retirer `preparer`, `receipt`, `receipt_fields` et leurs tests ; retirer `rebuild` et `rejected_sha256`, ou les brancher dans `load()` si un moteur refusé doit vraiment se refaire ; retirer `has_owned_stages` ; faire passer `write_atomic` par `core`. Garder les valeurs de l'énumération si elles entrent dans des clés déjà publiées (elles n'y entrent pas : seul `TensorRt` sert). | Lu (`grep` : aucun appelant hors `tests/unit/prepare`) |

## Défauts connus revérifiés

| ID | Statut | Note |
|---|---|---|
| TRT-05 | Toujours présent | `src/prepare/builder.h:35` : 20 min fixes, `tensorrt_backend.cpp:296-312` ne les change pas. Un délai dépassé est un `BuilderTimeout` sans trace : rien ne retient l'échec, chaque ouverture relance la même construction de 20 min. Le délai court pour chaque morceau (six pour RIFE fusionné, quatre pour DistilDRBA). Correction possible : délai proportionnel au temps attendu (les journaux `*.build.log` le donnent déjà) et, après un dépassement, ne plus réessayer cette clé sans le dire. Voir aussi A05-02. |
| TRT-17 | En partie présent | Pour cette zone : « reconstruits à chaque pilote » est à nuancer. Le jeton `driver` vient de `cudaDriverGetVersion` (`tensorrt_backend.cpp:305`, `identity.cpp:162`), la version de l'API CUDA du pilote (13000), pas le numéro du pilote : la clé change à chaque saut de version CUDA du pilote (quelques fois par an), pas à chaque mise à jour. Un moteur TensorRT n'en dépend pas ; on le garde pour rester identique aux clés du fork. « Sans purge » est confirmé : aucun code ne retire un moteur, son `.sha256`, son `.key-lock`, son `.build.log` ni son `.layers.json` ; chaque saut de pilote, chaque version de TensorRT et chaque taille de vidéo ajoutent des fichiers pour toujours. Les autres points de TRT-17 (`nvidia-smi`, carte inconnue en « ptx », doc MPEG-4, largeur 16/9) sont hors de cette zone : renvoyés. |
| TRT-18 | Réfuté pour cette zone | Les dossiers de travail du cache sont tirés au hasard (`cache_io.cpp:122-138`), marqués à leur propriétaire et refusés s'ils sont des liens symboliques (`:140-149`). Les noms fixes `timing.cache`, `layers.json`, `model.engine` sont à l'intérieur de ce dossier. Le défaut vise d'autres zones (fiches 18 et 25). |
| SEC-20 | Toujours présent | `src/gpu/cuda/backend/CMakeLists.txt:95` (zone 27) : `AJP_TRTEXEC_NO_FALLBACK` en `Release` seulement. Conséquence dans cette zone : A05-05. |
| QUA-03 | Toujours présent | Voir A05-08. |

## Renvois

| Vers l'audit | Où | Constat |
|---|---|---|
| 27 | `src/gpu/cuda/backend/CMakeLists.txt:95` | Repli `trtexec` gardé en `RelWithDebInfo`, le type des préréglages `linux-prefix-cuda-release` et `-machine` (SEC-20) |
| 01 | `src/gpu/cuda/backend/entry.cpp:56-59` | `pick_trtexec` appelé une fois à la création du backend : l'empreinte n'est plus revérifiée ensuite (A05-05) |
| 06 | `src/app/play/engine_install.cpp:130-158` | TRT-17 : échec de `nvidia-smi` gardé pour le processus, carte inconnue rangée en « ptx » |
| 18 | `src/app/play/crash_dump.cpp:90` | Le gestionnaire de plantage ne tue pas le `trtexec` en cours (A05-01) |
| 07 | appelants de `Backend::build` | Une construction par taille de source (formes fixes) : coût d'une taille nouvelle à mesurer (SPAN, plus six morceaux RIFE, plus quatre DistilDRBA) et taille du cache sans purge (TRT-17) |
| 36 | `src/prepare/identity.cpp:161-164` | Le jeton `driver` dans la clé : décider si la compatibilité avec les clés du fork vaut une reconstruction complète à chaque saut de CUDA du pilote |

## Points positifs

- Publication d'un moteur complète ou absente : contenu poussé sur le disque, renommage, dossier poussé
  (`cache_io.cpp:92-99`) ; empreinte revérifiée à la reprise (`cache.cpp:48-57`), testée (`cache_test.cpp:259`).
- Verrou par clé qui ne se perd pas : le fichier n'est jamais supprimé, le noyau rend le verrou à la mort du
  processus ; testé entre huit threads et quatre processus.
- Ordre des verrous fixe (clé, puis GPU) : pas d'interblocage possible entre deux constructions.
- `trtexec` lancé sans shell ni recherche dans `PATH`, ligne de commande Windows en UTF-16 avec la citation de
  `CommandLineToArgvW` testée, seules les poignées voulues héritées, groupe tué en entier à l'annulation.
- Timing cache travaillé sur une copie et publié seulement après succès, sous le verrou du GPU.
- Moteur désérialisé avant publication (`EngineCheck`) et ONNX rehaché après la construction.
- La clé couvre le contenu ONNX et ses poids externes, les réglages exacts (formes et précision comprises), la
  version de TensorRT, de CUDA et l'appareil (nom, SM, UUID).

## Questions ouvertes

- Faut-il garder le jeton du pilote dans la clé pour rester compatible avec les moteurs du fork, ou l'en retirer
  pour éviter une reconstruction complète à chaque saut de CUDA ? Le fork partage-t-il encore ce dossier ?
- Quelle politique de purge : retirer les moteurs d'une autre version de TensorRT et de CUDA au démarrage, ou
  plafonner la taille du cache ?
- Le délai de `trtexec` doit-il suivre le temps attendu mesuré sur la machine (TRT-05) plutôt que 20 min fixes ?
