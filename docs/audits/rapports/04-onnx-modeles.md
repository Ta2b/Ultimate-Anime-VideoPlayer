# Rapport de l'audit 04. Analyse et découpe des modèles ONNX

Branche : `audit/04-onnx-modeles`. Commit audité : `7739971`. Date : `2026-11-09`.

## Verdict

Les lecteurs protobuf de `prepare/` sont bornés et ont tenu 120 000 fichiers corrompus sous ASan et UBSan, et la
découpe de DistilDRBA vérifie toute sa colle ; mais un emplacement de poids en UTF-8 invalide fera planter le
lecteur sous Windows (à confirmer), la reconnaissance de RIFE laisse passer une partie de la colle sans la vérifier,
et rien dans la CI ne prouve qu'un modèle découpé calcule comme le modèle entier.

## Cadrage retenu

Réponses de Tatoub aux deux questions posées :

- **modèle fourni par `--model` ou `--rife-model`** : « jamais de plantage ». Un fichier tronqué ou hostile doit
  donner un refus clair ; un plantage ou une lecture hors bornes compte comme Majeur ;
- **découpe de RIFE et de DistilDRBA** : « seulement les modèles livrés » (RIFE 4.22 lite, DistilDRBA v2 lite).
  Une autre version n'a pas à se découper, mais elle doit être refusée proprement, jamais découpée de travers.

`onnx_blend` (interpolateur de Core ML) est hors cible : lu seulement pour la validité du modèle écrit.

Le fil tourne sans GPU ni Catch2. Les fichiers de `prepare/` de la fiche, avec `gpu/ir/protobuf.cpp`,
`core/sha256_blocks.cpp`, `core/file_sync.cpp`, `prepare/identity.cpp`, `prepare/sha256.cpp` et
`prepare/trtexec_command.cpp`, ont été compilés avec g++ 13 sous `-fsanitize=address,undefined`, et un petit
programme les a exercés :

- 6 × 20 000 mutations (octets changés, troncatures) du modèle de `blend_model`, et 300 du vrai
  `distil_drba_v2_lite.onnx` (copie des fichiers du projet), passées à `scan_onnx`, `read_onnx_graph`,
  `onnx_nodes`, `analyze_rife`, `analyze_drba`, `with_fixed_dims`, `with_outputs` et `extract_subgraph` : aucune
  erreur d'ASan ni d'UBSan ;
- le vrai DistilDRBA est reconnu (t = 4) et se coupe en quatre morceaux à 192 × 320 (7, 37, 37, 37 nœuds) ;
- les défauts A04-02 et A04-05 sont démontrés par ce programme. Les tests Catch2 qui échouent sont décrits.

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A04-01 | Majeur | `src/prepare/onnx_scan.cpp:165-168`, `src/prepare/onnx_scan.cpp:294` | L'emplacement d'un poids externe est lu tel quel dans l'ONNX (seul `\0` est refusé), puis `is_safe_location` le passe à `path_from_utf8`. Sous MSVC, construire un `std::filesystem::path` depuis un `u8string` qui n'est pas de l'UTF-8 valide lève `std::system_error` (conversion vers UTF-16 avec `MB_ERR_INVALID_CHARS`). Rien ne l'attrape (`main.cpp` n'a pas de `catch` global) : `std::terminate`. Le module a pourtant `is_utf8` (`utf8.h`). | Sous Windows 11, un `--model` ou `--rife-model` dont un `external_data.location` vaut l'octet `0xFF` (fichier abîmé ou fabriqué) : `read_model` appelle `scan_onnx`, puis `is_safe_location`, et le lecteur se ferme sans message au lieu de refuser le modèle. Sous CachyOS, le même fichier est refusé plus loin (poids introuvable). | Dans `LocationWalker::walk`, refuser `!is_utf8(location)` comme un emplacement invalide (`ModelUnreadable`) ; `is_safe_location` peut aussi rendre `false` sur un texte qui n'est pas de l'UTF-8. | Lu. À confirmer sous Windows : un ONNX de quelques octets (ModelProto → graphe → initialiseur → `external_data {key: "location", value: "\xff"}`, `data_location = 1`) passé à `ajp --model=…`. Test qui échoue aujourd'hui sur tous les systèmes, à ajouter à `tests/unit/prepare/onnx_scan_test.cpp` : `CHECK_FALSE(onnx_external_locations(modele_avec_emplacement("\xff")))`. |
| A04-02 | Mineur | `src/prepare/onnx_split.cpp:125`, `src/prepare/onnx_split.cpp:129` | `elements *= dim` est un produit en `int64_t` signé fait avant le plafond `kElementCap` (2^40) : avec une dimension au-delà de 2^23, le produit déborde (comportement indéfini). La vérification `raw.size() == count * 4` qui suit empêche toute lecture hors bornes. | Un initialiseur de dimensions 2^40 × 2^30 dans un RIFE ou un DistilDRBA fourni (`read_onnx_graph`, appelé par `analyze_rife` et `analyze_drba`). Un compilateur peut supposer que le débordement n'arrive pas et retirer le plafond. | Plafonner avant de multiplier : `elements = value == 0 ? 0 : (value > kElementCap / elements ? kElementCap : elements * value)`, en `std::uint64_t`. | Démontré : UBSan, `onnx_split.cpp:129:18: runtime error: signed integer overflow: 1073741824 * 1099511627776`. Test : le même tenseur lu par `read_onnx_graph` dans un test construit avec `-fsanitize=undefined` (la tâche de CI qui en a un). |
| A04-03 | Mineur | `src/prepare/rife_split.cpp:210-231`, `src/prepare/rife_split.h:4-6` | L'en-tête promet que « toute la structure est reconnue », mais `analyze_rife` ne vérifie que des noms aux frontières, des décomptes d'opérateurs et quelques constantes. Les entrées 0 et 1 du Concat de chaque bloc (images et encodages déformés par le flot), l'entrée 1 du Concat du bloc 0, les entrées des GridSample, l'addition du flot d'un bloc au suivant et le mélange final ne sont pas comparés au câblage attendu. Ce sont justement les opérations que les noyaux refont. DistilDRBA, lui, vérifie chaque nœud de sa colle (`drba_expected_wiring`). | Un RIFE fourni par `--rife-model` (un autre export de vs-mlrt, 4.18 ou 4.26) qui garde les mêmes noms de nœuds mais range autrement les canaux du Concat, ou déforme les images au lieu des encodages : la découpe est acceptée, le chemin fusionné calcule une autre chose que le réseau, sans refus ni message. La lecture montre des images intermédiaires fausses. | Écrire `rife_expected_wiring(blocks, reduce)` sur le modèle de `drba_expected_wiring` (nœuds de la colle, entrées exactes, constantes), et le comparer comme `analyze_drba` le fait. | Lu (comparaison avec `drba_split.cpp:305-321`). |
| A04-04 | Mineur | `src/prepare/rife_split.cpp:381`, `src/prepare/drba_split.cpp:499`, `src/prepare/rife_split.cpp:392`, `src/prepare/drba_split.cpp:508` | Le nom des morceaux dérivés part de `stem().string()`, qui passe par la page de code ANSI sous Windows et lève une exception devant un caractère qu'elle n'a pas ; les noms relus du manifeste repassent par `path(std::string)` (`dir / piece`), qui lit les octets dans cette même page. Ce sont aussi deux des `path::string()` que `tools/ci/check-path-string.sh` veut voir diminuer. | Sous Windows 11, un `--rife-model` nommé `リゼロ_rife.onnx` sur un système en page 1252 : `derive_rife_fused` lève `std::system_error`, rien ne l'attrape, le lecteur se ferme au moment d'ouvrir l'interpolation. Sous CachyOS, rien. | `core::to_utf8(rife_onnx.stem())` pour le nom, `path_from_utf8(piece)` pour relire un morceau (ici et dans les quatre backends qui lisent le manifeste). | Lu. À confirmer sous Windows avec un RIFE renommé en japonais. |
| A04-05 | Mineur | `src/prepare/model_source.cpp:36-43`, `src/prepare/model_source.cpp:87` | `escapes_folder` compare `weakly_canonical(dossier)` au chemin du poids. Pour un modèle donné sans dossier (`--model=modele.onnx`, gardé tel quel par `find_file`, `options.cpp:616`), `parent_path()` est vide, `weakly_canonical("")` aussi, et la comparaison ne voit jamais de sortie : un poids externe qui est un lien symbolique vers l'extérieur passe. | Un modèle extrait d'une archive dans le dossier courant, avec `poids.bin -> /ailleurs/…` : la vérification est sautée, `onnx_source_key` hache le fichier pointé et trtexec le lit. Avec `--model=./modele.onnx`, le même lien est refusé. | Faire `std::filesystem::absolute(model.onnx_path)` avant de prendre `parent_path()` (dans `read_model`), ou traiter une racine vide comme le dossier courant. | Démontré : avec un lien `poids.bin -> /etc/hostname`, `escapes_folder("", "poids.bin")` rend `false`, `escapes_folder(".", "poids.bin")` rend `true`. Test : le même cas dans `tests/unit/prepare/builder_test.cpp`, modèle ouvert par un chemin relatif sans dossier. |
| A04-06 | Mineur | `src/prepare/model_source.cpp:71-111` | `read_model`, qui vérifie un modèle avant de lancer trtexec, regarde son entrée (nom, type, rang, dimensions fixes) mais jamais ses sorties : ni leur nombre, ni le facteur d'agrandissement, ni le nombre de plans. | `--model=4x_quelquechose.onnx` (ou un ×1, ou un modèle à sortie RGBA) en FP16 : la construction TensorRT tourne plusieurs minutes avant que l'erreur arrive à l'exécution, ou plus loin selon ce que vérifie l'étage. | Lire les sorties dans `scan_onnx` (comme les entrées) et refuser avant la construction une sortie qui n'est pas 1 × 3 × 2H × 2W pour un agrandisseur (les dimensions symboliques de la sortie suffisent à reconnaître « 2*H »). | Lu. À confirmer : `ajp --model=un_modele_x4_fp16.onnx video.mkv` sur la 4070 Ti, et regarder si le refus arrive avant ou après la construction. Voir le renvoi vers 09. |
| A04-07 | Mineur | `src/prepare/rife_split.h:78`, `src/prepare/drba_split.h:104`, `src/prepare/rife_split.cpp:405-409` | Les morceaux dérivés sont gardés sous un nom qui ne porte que `fusion1`, la taille et l'empreinte du modèle source ; un morceau déjà présent n'est jamais réécrit. Changer `extract_subgraph`, `analyze_*` ou `value_info` sans penser à monter `fusion1` laisse les anciens morceaux en place, et les moteurs construits dessus. Aucun test ne relie la version au contenu produit. | Une correction de la découpe (A04-03 par exemple) est livrée : chez Tatoub, les morceaux de la version précédente restent dans `derived_models` et le chemin fusionné garde l'ancien comportement, alors qu'une installation neuve a le nouveau. | Mettre dans le nom une empreinte du code de découpe, ou un test qui hache les morceaux d'un graphe fabriqué et échoue si l'empreinte change sans que `kRifeFusedDerivation` ou `kDrbaFusedDerivation` change. | Lu. |
| A04-08 | Mineur | `tests/unit/prepare/rife_split_test.cpp:461-499`, `tests/unit/prepare/drba_split_test.cpp:387-427`, `tests/unit/prepare/drba_split_test.cpp:138`, `tests/fuzz/fuzz_parsers.cpp:33-37` | Rien dans la CI ne montre que la découpe garde le calcul du réseau entier : les tests sur les vrais modèles sont sautés quand `~/ajp/models/rife` manque (toujours en CI, aucun workflow n'y met les modèles), les comparaisons numériques sont cachées (`[.rife_fused]`, `[.drba_fused]`) et demandent un GPU, et le graphe de DistilDRBA des tests est fabriqué à partir de `drba_expected_wiring` lui-même. Côté entrées hostiles, la cible de fuzz `onnx` n'appelle que `gpu::ir::load_onnx` : aucun lecteur de `prepare/` n'est fuzzé (SEC-21). | Une régression de `extract_subgraph` ou de la reconnaissance passe toutes les tâches de CI ; elle ne se voit qu'en lançant à la main les tests cachés sur la machine de Tatoub. | Faire télécharger les deux modèles livrés (`tools/release/fetch-models.sh`, avec cache) par une tâche de CI et y lancer les tests du catalogue ; ajouter une comparaison sur CPU à petite taille (l'interpréteur `gpu/ir/interpreter.cpp` exécute le modèle entier et les morceaux, la référence CPU des noyaux fait la colle). Ajouter une cible de fuzz `prepare_onnx` (scan, graphe, découpe, `with_*`) avec une graine qui est un vrai petit modèle (`blend_model`). | Lu. |
| A04-09 | Mineur | `src/prepare/onnx_metadata.cpp`, `src/prepare/onnx_metadata.h:2`, `tests/unit/prepare/onnx_metadata_test.cpp` | Code mort : `onnx_metadata` n'a plus d'appelant depuis le retrait de GMFSS (`b2f52a8`, qui lisait `kGmfssExportKey`). L'en-tête dit encore que « le lecteur la consulte à chaque recherche de modèle ». Le fichier porte 7 des `path::string()` comptés par la CI, qui lèveraient sous Windows. | Aucun pour l'utilisateur ; un lecteur du code le croit en service et le maintient. | Retirer les deux fichiers, leur test et la ligne de `CMakeLists.txt`, et baisser le plafond de `check-path-string.sh` d'autant. Le commentaire de `gpu/ir/protobuf.h:39` qui le cite aussi. | Lu (`git grep onnx_metadata` hors du module : rien). |
| A04-10 | Mineur | `src/prepare/onnx_scan.cpp:33-91`, `src/prepare/onnx_scan.cpp:42` | Un second décodeur du format filaire, alors que `protobuf_fields.h:26` et `gpu/ir/protobuf.h:40` disent que celui de `gpu/ir` est le seul (P1.31). Ils divergent : celui-ci tronque le numéro de champ en 32 bits (`tag >> 3` vers `uint32_t`) là où l'autre refuse au-delà de 2^29, et accepte un dixième octet de varint quelconque. | Un fichier fabriqué dont une étiquette porte le champ 2^32 + 7 : `scan_onnx` le prend pour le graphe (champ 7), les autres lecteurs le refusent. Pas de plantage, mais deux lectures du même fichier qui ne s'accordent pas, sur le chemin qui vérifie les poids externes. | Faire passer `onnx_scan.cpp` par `protobuf::Fields`, comme les autres fichiers du module. | Lu. |
| A04-11 | Mineur | `src/prepare/rife_split.cpp:30-90`, `src/prepare/drba_split.cpp:52-95`, `src/prepare/rife_split.cpp:296-433`, `src/prepare/drba_split.cpp:420-533`, `src/prepare/onnx_split.cpp:168`, `src/prepare/onnx_outputs.cpp:51`, `src/prepare/onnx_dims.cpp:128-137` | Doublons : `Index`, `text_is`, `same`, `block_name`, `exists_nonempty`, la lecture des nombres du manifeste et toute la fonction `derive_*` sont recopiés entre RIFE et DistilDRBA ; `graph_of` existe deux fois, et une troisième en ligne dans `with_fixed_dims`. | Une correction faite d'un côté seulement (A04-04, A04-07) laisse l'autre faux. | Un `split_common.h` interne (index, manifeste, dérivation paramétrée par l'analyse et la découpe) et un seul `graph_of` dans `protobuf_fields.h`. | Lu. |

## Défauts connus revérifiés

La fiche n'en cite aucun et `SYNTHESE.md` n'a pas de renvoi vers 04. La fiche nomme SEC-21 parmi les points :

| ID | Statut | Note |
|---|---|---|
| SEC-21 | Toujours présent | `tests/fuzz/corpus/onnx/seed.bin` fait toujours 16 octets et `tests/fuzz/fuzz_parsers.cpp:33-37` ne fuzze que `gpu::ir::load_onnx` ; `scan_onnx` et les autres lecteurs de `prepare/` ne le sont pas (A04-08). Les lecteurs eux-mêmes sont bornés : 120 000 mutations sous ASan et UBSan sans erreur, profondeur limitée à 64 dans `LocationWalker`, taille limitée à 2 Gio par `read_onnx_bytes`. |

## Renvois

| Vers l'audit | Où | Constat |
|---|---|---|
| 09 | `src/app/play/derived_model.cpp:19` | Même défaut que A04-04 : `source.stem().string()` dans le nom des modèles dérivés (mouvement de RIFE et de DistilDRBA, DistilDRBA à taille fixe). |
| 09 | `src/app/play/motion_model.cpp:90`, `src/app/play/motion_model.cpp:121` | `filename().string()` et `stem().string()` sur le chemin du RIFE (même risque sous Windows). |
| 09 | `src/app/play/upscale_stage.cpp:198` | L'étage suppose `kSpanFactor` ; vérifier ce qui se passe quand le moteur d'un `--model` sort une autre taille (voir A04-06). |
| 28 | `tests/fuzz/fuzz_parsers.cpp:33-37`, `.github/workflows/fuzz.yml` | Aucune cible de fuzz pour les lecteurs ONNX de `prepare/` ; graine ONNX de 16 octets (SEC-21). |
| 05 | `src/prepare/builder.cpp:174`, `src/prepare/builder.cpp:244` | `read_model` relit et rehache tout l'ONNX et ses poids avant et après chaque construction : coût négligeable pour SPAN, à garder en tête pour un modèle proche de 2 Gio. |

## Points positifs

- Chaque lecture protobuf vérifie ses bornes : longueur d'un champ contre la fin du message, varint de dix octets au
  plus, groupes refusés ; aucune erreur d'ASan ni d'UBSan sur 120 000 fichiers corrompus.
- `read_onnx_bytes` prend la taille au système de fichiers, refuse un dossier, un fichier vide ou de plus de
  2 Gio, et détecte un fichier raccourci pendant sa lecture.
- Les poids externes sont cherchés partout (graphes imbriqués, tenseurs creux, fonctions, entraînement), un
  emplacement absolu, avec `..`, `\` ou `:` est refusé, et les liens symboliques sont suivis (hors A04-05).
- `extract_subgraph` refuse tout tenseur qui n'est ni entrée du morceau, ni constante, ni produit, et refuse poids
  externes, initialiseurs creux et graphes imbriqués plutôt que de les recopier à moitié.
- La reconnaissance de DistilDRBA compare chaque nœud de la colle, ses entrées et ses constantes au graphe de
  référence, et le vrai `distil_drba_v2_lite.onnx` la passe.
- Un refus de la découpe garde le modèle entier (`player_engines.cpp`) : l'utilisateur garde l'interpolation.
- Les morceaux et les manifestes s'écrivent de façon atomique, le manifeste en dernier ; un nom de morceau du
  manifeste ne peut pas contenir de dossier.
- La partie Windows des chemins est déjà pensée dans `is_safe_location` (`has_root_path`, `C:poids.bin`).

## Questions ouvertes

- Sous TensorRT, un `--model` en FP32 est refusé (`check_precision` : « entrée de type ONNX 1 au lieu de 10 »).
  Est-ce voulu, ou faut-il l'accepter (TensorRT le construit très bien) ou le convertir ? Le message est-il montré
  tel quel à l'utilisateur (audit 24) ?
- Sous Windows, un emplacement de poids `CON`, `NUL` ou `aux.bin` passe `is_safe_location` ; selon la version de
  Windows, il désigne un périphérique. À confirmer sur la machine Windows de Tatoub, sans urgence.
- RIFE 4.25 se découpe toujours (5 blocs) alors que le lecteur ne livre plus que 4.22 lite : faut-il garder ce
  chemin, ou refuser tout ce qui n'est pas 4.22 lite (plus simple que A04-03) ?
