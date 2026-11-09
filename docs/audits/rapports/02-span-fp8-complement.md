# Complément de l'audit 02. SPAN FP8 : les renvois reçus

Branche : `claude/project-thread-xvel3y`. Commit audité : `65f366b`. Date : `2026-11-09`.

## Verdict

Des quatre renvois reçus, un ouvre un défaut neuf et mineur (A02-14 : la paire paquet et reçu s'écrit sans fsync,
et un paquet abîmé par une coupure est pris pour un refus de la porte, que le lecteur ne refait plus seul). Deux
sont des doublons (A01-02, A37-01), et un est réfuté : le reçu porte bien le SHA du paquet dérivé, et le lecteur le
vérifie.

## Cadrage retenu

Consigne du coordinateur : vérifier seulement les renvois de `audits-docs/complements/02.md` dans le code, conclure
chacun par confirmé, réfuté ou doublon, et signaler toute baisse de qualité d'image faite pour la vitesse sans
mesure. Aucune question posée à Tatoub. Le fil tourne sans GPU : tout est lu sur `65f366b`, où les numéros de ligne
des renvois tombent encore juste.

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A02-14 | Mineur | `src/app/play/span_derive.cpp:27-35`, `:49-60`, `:111-138`, `src/app/play/span_int8.cpp:113`, `:117`, `:152-155`, `src/app/play/span_derive.cpp:418-420` | `commit_derivation` écrit le paquet `.span8` et son reçu sous `.part` puis les renomme, sans `sync_file` ni `sync_directory` (la règle P5.11 de `src/core/file_sync.h`). `Swap::install` déplace d'abord l'ancien fichier en `.old` : le renommage se fait donc sur un nom libre, et l'heuristique d'ext4 (`auto_da_alloc`), qui force l'écriture quand un renommage remplace un fichier, ne s'applique pas. Une coupure peut laisser un paquet ou un reçu vide sous son nom définitif. Au démarrage suivant, `find_span_variant` rend un refus ordinaire, pas `stale` (« receipt unreadable », « unknown format » pour un reçu vide, qui se lit comme une table vide, ou « changed since its receipt » pour un paquet vide). `make_span_prepare` pose alors `refused = true`, et `should_start_span_auto` ne relance plus la préparation. C'est une instance de QUA-05, dans le même genre qu'A45-07 pour les artefacts Vulkan. | RTX 4070 Ti, cran Performance : la préparation silencieuse finit, la machine s'éteint quelques secondes plus tard (coupure, batterie vide). À la vidéo suivante, SPAN joue en TensorRT FP16 (environ 14,3 ms au lieu de 9,3 ms), et le menu annonce « Rejected: … » comme si la porte de qualité avait refusé le paquet. Rien ne se refait sans un clic sur « Prepare again for this video ». L'image ne se dégrade pas (FP16), mais le gain du FP8 est perdu sans raison visible. Les `.old` d'une opération interrompue entre deux renommages restent aussi sur le disque, puisque seul le processus qui les a créés les retire. | Appeler `core::sync_file` sur chaque `.part` avant `install`, puis `core::sync_directory` sur le dossier après les renommages. Dans `find_span_variant`, classer `stale` un reçu illisible, un reçu de format vide et un paquet dont le SHA ne correspond plus : ces cas ne viennent pas d'un jugement de la porte, et la préparation automatique doit les refaire. Purger les `*.span8.old` et `*.toml.old` de `models_dir` au début d'une préparation. Test : un reçu vide et un paquet tronqué donnent `stale`, pas `refused`. | Lu. Pour la coupure, à confirmer : ext4 `data=ordered`, préparation suivie d'un `echo b > /proc/sysrq-trigger`, puis taille du `.span8` et du `.toml`. |

## Renvois traités

| # | De | Où | Conclusion | Preuve et note |
|---|---|---|---|---|
| 1 | 01 | `src/gpu/cuda/backend/tensorrt_span_fp8.cpp:78-89` | Doublon d'**A01-02** | Confirmé dans le code. `order(reads, writes)` (`:78`) ouvre l'opération de rang M, et le corps (`enqueue_body`, `:85`) lui appartient. `begin_slot` (`:89`) appelle alors `join()` (`tensorrt_kernels.cpp:305-306`), et `apply` pose l'événement du rang M (`tensorrt_lanes.cpp:105`). La queue (`enqueue_tail`, `:97`) écrit ensuite `outputs` et la surface de l'anneau hors du rang M. Une autre voie qui lit `outputs` en attendant le rang M peut lire une sortie à moitié écrite à la première image d'une nouvelle génération. La correction d'A01-02 (jointure avant `order()`) vaut ici. `slot()` met `generation_` à jour (`tensorrt_kernels.cpp:255`) : la jointure avancée doit le faire aussi, sinon `begin_slot` joint une seconde fois. L'attente de l'emplacement doit rester après le corps (commentaire `:87-88`), seule la jointure remonte. Aucun effet sur la qualité en dehors de cette image. |
| 2 | 20 | `src/app/play/span_derive.cpp:27-35` | **Confirmé** : A02-14 | `write_file` passe par `ofstream` et `close`, sans `sync_file`. Les renommages de `Swap::install` (`:49-60`) se font sans `sync_directory`. Le commentaire `:124` (« une coupure laisse l'ancien état entier ») ne tient que pour une coupure avant les renommages. |
| 3 | 36 | `src/app/play/span_int8.cpp:114-125` | **Réfuté** | Le reçu porte le SHA du paquet dérivé. `span_derive.cpp:258-259` le calcule sur les octets écrits, `:267` le range dans `receipt.model_sha256` (que `derive.cpp:77` écrit, avec le commentaire « du paquet ".span8" écrit », `derive.h:85`). `find_span_variant` le relit et le compare au fichier (`span_int8.cpp:152-155`) avant d'accepter la variante. Les deux outils Python l'écrivent aussi (`tools/span/derive_span_fp8.py:307`, `tools/span/quantize_span.py:392`). Un paquet tronqué n'est donc jamais joué. De plus, `span_fp8::parse` le refuse au chargement (CRC-32). Le vrai défaut est ailleurs : ce refus n'est pas refait automatiquement (A02-14). La ligne de QUA-05 dans le rapport 36 (« un paquet FP8 tronqué avec un reçu "passed" n'est pas reconnu ») est à corriger dans ce sens. |
| 4 | 37 | `src/gpu/cuda/backend/span_fp8_kernels.cuh:241` | Doublon d'**A37-01** (qui cite déjà cette ligne) | Le tag n'est pas faux en soi. `cutlass::arch::Sm89` ne fait que choisir la boucle multi-étages de type SM80 (cp.async et `mma.sync`). Les gardes de CUTLASS activent le `mma` FP8 m16n8k32 et le `mma` FP16 dès `__CUDA_ARCH__ >= 890` (`third_party/cutlass/include/cutlass/arch/mma_sm89.h:56-64`, `mma_sm80.h:51-52`), donc aussi sur sm_120, où l'instruction existe. Les noyaux maison n'ont pas de garde d'architecture (`span_fp8_kernels.cuh:52`, `:62`), et la mémoire partagée maximale par bloc est de 99 Ko sur sm 8.9 comme sur sm 12.0. Ce qui manque, c'est un essai : rien n'a encore tourné sur une RTX 50 (`docs/span-fp8-cutlass.md:211-217`), et la porte, aveugle aux NaN (A02-01), est la seule garde. Baisse de qualité pour la vitesse : sur sm_120, le FP8 remplace le FP16 sans aucune mesure de gain, déjà signalé par A37-01. Le gain mesuré, 9,3 ms contre 14,3 ms en 1088×1920, ne vaut que pour la RTX 4070 Ti. |

## Points positifs

- Le reçu v2 lie le paquet par son SHA, et `find_span_variant` vérifie ce SHA avant toute autre décision : un
  paquet modifié ou tronqué n'est jamais joué.
- L'échange par `.part` et `.old` remet l'ancien état en place en cas d'échec pendant l'opération (`Swap::rollback`) :
  seule une coupure de courant échappe à cette garde.
