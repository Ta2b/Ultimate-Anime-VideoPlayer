# Rapport de l'audit 36. État persistant et compatibilité entre versions

Branche : `claude/project-thread-ggzwid`. Commit audité : `58ddedf0` (le code est celui de `2748cdeb`, seuls des
rapports ont changé entre les deux). Date : `2026-11-09`.

## Verdict

L'écriture atomique commune est saine (fsync du temporaire puis du dossier), mais le passage d'une version à
l'autre n'est pas tenu : un retour arrière efface toutes les clés de `play.toml` écrites par la version plus
récente dès qu'une seule valeur lui est inconnue (démontré), la calibration de la machine n'est jamais invalidée
par une mise à jour du lecteur, du module TensorRT ou du pilote (l'Auto calibré reste bridé à l'image d'une
version plus lente), et le paquet TensorRT de 1,3 Go est installé sans fsync puis jugé « installé » sur la seule
présence de son module.

## Cadrage retenu

Pas de questions à Tatoub (consigne du coordinateur) ; options retenues :

- cible : NVIDIA sous TensorRT, Windows 11 et CachyOS ; Vulkan seulement comme repli sur une RTX ;
- parcours prioritaires : mise à jour automatique puis retour arrière par la garde (renvoi 26), coupure de
  courant pendant ou juste après une écriture, profil itinérant sous Windows, désinstallation ;
- toute perte de qualité d'image qui survit à une mise à jour ou à un retour arrière compte comme un défaut ;
- RIFE 4.25, RIFE 4.22 lite et DistilDRBA comptent autant. Côté état persistant, rien n'empêche DistilDRBA d'être
  rapide en plus des défauts déjà connus (calibration et Auto qui ne l'essaient pas, A25-08 et A29-05), sauf
  A36-04 : ses modèles dérivés par taille ne se refont jamais d'une version à l'autre ;
- le fil tourne sans GPU ni `~/ajp-deps` : les tests du dépôt n'ont pas été compilés. A36-01 est démontré par
  `settings_toml_lenient.cpp` compilé tel quel avec toml++ 3.4 et un lecteur réduit.

### Inventaire des fichiers persistants

| Fichier | Emplacement Windows / CachyOS | Format, version | Écriture | Lecture d'un fichier abîmé | Nettoyage |
|---|---|---|---|---|---|
| `play.toml` | `%APPDATA%\ajp\ajp` / `~/.local/share/ajp/ajp` | TOML, `version = 1` jamais lue pour migrer | atomique (`settings_toml.cpp:626`) | lecture lâche, sinon mis de côté (`player.cpp:180`) | — |
| `settings.toml` | idem | TOML, version exigée égale (`toml_fields.h:106`) | atomique | refusé entier, mis de côté, assistant relancé (`player.cpp:262-267`) | — |
| `user.toml` | idem | TOML (`[keys]`) | atomique (`user_overlay.cpp:255`) | A24-07 | — |
| `update.toml` | idem | TOML, sans version | atomique, quatre écrivains (A26-10) | défauts | — |
| `calibration.toml` | idem | TOML, `version = 1` jamais lue | atomique (`machine_calibration.cpp:257`) | ignoré | jamais invalidé (A36-02) |
| `bench.toml` | idem | TOML, pilote noté et vérifié (`player.cpp:403-404`) | atomique | ignoré | — |
| `resume.tsv`, `auto_memory.toml` | idem | lignes tolérantes, bornées (200) | atomique, toutes les 15 s / au palier tenu 6 s | lignes illisibles ignorées | bornés |
| `ajp.log*`, `crashes/*.dmp` | idem (itinérant) | texte, minidumps | spdlog, vidé chaque seconde | — | 4 lancements, 5 dumps (WIN-03) |
| `updates/<v>/`, `updates/previous/` | idem (itinérant) | archive, copie entière de l'installation | copie fichier par fichier | — | A26-06, `previous` jamais retiré sans retour arrière |
| `engines/aji-v3-*.engine` (+ `.sha256`, `.key-lock`, `.build.log`), `aji-timing-*.cache` | `%LOCALAPPDATA%\ajp\ajp` / `~/.local/share/ajp/ajp` | TensorRT, clé SHA-256 opaque | atomique avec fsync (`cache_io.cpp:91-98`) | SHA-256 vérifié (`cache.cpp:48-57`) | aucun (A33-04) |
| `engines/` Vulkan (`net_build.cpp:194`) | idem | artefact versionné | renommage **sans fsync** | `deserialize` + `check_artifact`, refait | aucun |
| `backends/tensorrt/<version>/` | idem | paquet 1,3 Go | renommage de dossier **sans fsync** (A36-03) | présence seule | aucun (A36-06) |
| `modeles/*.onnx` (dérivés) | idem | nom = source + suffixe + SHA de la source | atomique | taille > 0 seulement | aucun (A09-11), jamais refaits (A36-04) |
| `modeles/` FP8 et INT8 + reçus (`span_derive.cpp:111`) | idem | paire paquet / reçu | renommages **sans fsync** | reçu sans SHA du paquet | — |
| `plans/*.shots` | idem | binaire à signature | renommage **sans fsync** (`shot_cache.cpp:139-145`) | signature et bornes, ignoré | borné (`trim`) |
| `shaders.cache` | idem | `pl_cache` | renommage **sans fsync** (`placebo_open.cpp:299-305`) | refusé par `pl_cache_load` | `.part` jamais purgés (A10-11) |
| temporaires | `%TEMP%` / `/tmp` : `ajp-calibration`, `ajp-bench`, `ajp-compare`, `ajp-update-*` | — | — | — | retirés en fin de tâche, pas après un arrêt brutal |
| registre, `.desktop`, MIME | HKCU / `~/.local/share/applications` | — | `--register` | — | `--unregister` jamais lancé à la désinstallation (renvoi 19→28) |

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A36-01 | Majeur | `src/app/play/settings_toml_lenient.cpp:73-74`, `src/app/play/settings_toml.cpp:614-618`, `:403` | Au retour arrière, la version plus ancienne réécrit `play.toml` sans **aucune** des clés de la plus récente dès qu'une seule valeur d'une clé connue lui est inconnue. `keep_unknown_keys` commence par lire le fichier existant avec le lecteur strict (`settings_from_table`) : un refus (nouvelle valeur d'`interpolation.engine`, de `picture.scaler`, d'un cran de précision…) fait `return` avant de garder quoi que ce soit. La valeur inconnue elle-même revient au défaut (A22-06), et `version` est réécrit à 1 : la version récente, relancée ensuite, ne voit même plus qu'un ancien lecteur est passé par là. | 0.11 ajoute `interpolation.engine = "rife425_fused"` et une clé `image.sharpen_lines`. Elle plante trois fois sur la machine de Tatoub, la garde rend 0.10 (`main.cpp:398-401`). Le premier réglage touché (le volume) réécrit `play.toml` : le moteur revient à RIFE par défaut et `sharpen_lines` disparaît. La 0.11 corrigée s'installe : tous les réglages de qualité propres à 0.11 sont aux défauts, sans un mot. P2.22 promet le contraire. | Dans `keep_unknown_keys`, prendre pour base la lecture lâche (`parse_settings_lenient`) au lieu du lecteur strict, et garder telle quelle toute valeur que la lecture lâche a écartée (la clé connue à valeur inconnue fait partie de ce qui doit survivre). Ne jamais abaisser `version` : écrire `max(version lue, kSettingsVersion)`. | **Démontré** : `settings_toml_lenient.cpp` compilé tel quel (toml++ 3.4, `PlaySettings` réduit à `engine` et `work_height`). Fichier de « 0.11 » avec seulement des clés nouvelles : `sharpen_lines` et `scene_guard` gardés. Même fichier avec `engine = "rife425_fused"` : sortie `[image] work_height = 1440`, `[interpolation] engine = 'rife'`, plus aucune clé nouvelle, `version = 1`. Test à ajouter à `settings_test.cpp` (après le test P2.22, l. 620) : même fichier avec `[interpolation] engine = "inconnu"` ; après `save_settings`, `future_option = 7` et `engine = 'inconnu'` doivent rester (échoue aujourd'hui). |
| A36-02 | Majeur | `src/app/play/machine_calibration.cpp:240-252`, `:160-187` | `calibration.toml` n'est invalidé que par le nom du GPU, la hauteur de l'écran et la batterie. Ni la version du lecteur, ni celle du module TensorRT, ni le pilote, ni le moteur mesuré n'y sont écrits ou comparés (`version = 1` est écrit et jamais lu). `bench.toml`, lui, est refusé quand le pilote change (`player.cpp:403-404`). Une chaîne calibrée par une version lente reste donc la chaîne de l'Auto calibré pour toujours. | Calibration faite en 0.10.3 : le Rapide tient 1080p SPAN. 0.11 corrige les défauts de vitesse de RIFE et DistilDRBA fusionnés (audit 03) et active le FP8 sous Windows (A34-01). La carte tiendrait 1440p, mais l'Auto calibré garde 1080p : image plus petite à chaque lecture, sans indice que « Refaire le banc » changerait quelque chose. Même chose après un pilote plus rapide, ou dans l'autre sens (un pilote plus lent fait perdre des images). | Écrire dans `calibration.toml` la version du lecteur, le moteur et son jeton (`TrtRuntime::token()`, pilote compris), comme `bench.toml`. À la lecture, un écart rend la calibration « ancienne » : l'Auto s'en sert encore comme plancher, mais une notice propose de la refaire, ou la refait seul au prochain lancement sur secteur. | Lu. Gain à confirmer sur la machine de Tatoub : calibrer en 0.10.3, mettre à jour, puis comparer la hauteur de travail de l'Auto avec celle d'une calibration refaite. |
| A36-03 | Majeur | `src/app/play/engine_install.cpp:539-571`, `:69-71`, `src/app/play/engine_downloads.cpp:130` | Le paquet TensorRT (module, `libnvinfer`, `trtexec`, environ 1,3 Go) est extrait dans un dossier de travail puis mis en place par un renommage de dossier, sans aucun fsync des fichiers. `installed()` ne regarde ensuite que la présence du module, et seul un `AbiMismatch` déclenche une réinstallation. QUA-05 parlait d'un cache vide ; ici c'est le moteur lui-même. | CachyOS (btrfs, écriture différée jusqu'à 30 s) : Tatoub installe TensorRT depuis l'assistant, la coupure de courant (ou un gel du pilote suivi d'un arrêt forcé) arrive dans la demi-minute. Au redémarrage, `libnvinfer.so` ou le module font 0 octet mais existent : `dlopen` échoue, le lecteur passe à Vulkan, et ne réinstalle jamais. Plus de FP8, plus de fusion, chaîne calibrée plus petite, sans message qui dise pourquoi. | `core::sync_file` sur chaque fichier extrait et sur `kPackageNote` avant le renommage, puis `sync_directory` du parent. À l'ouverture, un échec de chargement autre qu'`AbiMismatch` sur un paquet installé relance la vérification des SHA-256 de ses fichiers (déjà connus de l'index) et réinstalle s'ils diffèrent. | Lu. À confirmer sur CachyOS : tronquer `backends/tensorrt/<version>/libnvinfer.so.*` à 0 octet, lancer, lire au journal le code du refus et vérifier qu'aucune réinstallation n'est proposée. |
| A36-04 | Mineur | `src/app/play/derived_model.cpp:18-22`, `src/app/play/drba_model.cpp:30-41`, `src/app/play/motion_model.cpp:97-100` | Le nom d'un modèle dérivé (`<source>-<suffixe>-<SHA de la source>.onnx`) ne porte pas la version de la transformation. Un fichier présent et non vide est repris tel quel. Une version qui corrige `with_motion_outputs`, `with_drba_motion_outputs` ou `with_fixed_dims` ne refait donc jamais les fichiers déjà dérivés, et la clé du moteur TensorRT (qui suit le contenu du dérivé) ne change pas non plus. Même cause qu'A04-07, pour les autres dérivés ; au retour arrière, l'ancienne version reprend aussi les dérivés de la nouvelle. | 0.11 corrige la sortie mouvement de DistilDRBA (A03-03). Chez Tatoub, `modeles/distil_drba_v2_lite-mouvement-<sha>.onnx` de 0.10 reste, le moteur construit dessus aussi : le correctif de qualité n'arrive jamais. De même pour chaque `distil_drba…-<l>x<h>.onnx` par taille. | Ajouter une version de transformation au nom (`-t2`) ou au SHA, par transformation ; la monter avec toute correction. Purger les dérivés d'une version de transformation inconnue (avec A09-11). | Lu. |
| A36-05 | Mineur | `src/app/play/updater.cpp:273-293` | `format_update_state` réécrit `update.toml` avec ses seuls champs connus. Au retour arrière, l'ancienne version efface les champs ajoutés par la nouvelle (par exemple une marque « application en cours », correction proposée pour SEC-07, ou un compteur de garde corrigé pour A26-09). Contrairement à `play.toml`, aucune clé inconnue n'est gardée. | 0.11 ajoute `applying = true` pour se protéger d'une coupure pendant l'application. Une 0.10 relancée par la garde réécrit le fichier : la marque disparaît, et la prochaine application coupée retombe dans SEC-07. | Relire la table existante et n'y remplacer que les champs connus (comme `core::format_settings` le fait pour `settings.toml`). | Lu. |
| A36-06 | Mineur | `src/core/model_index.cpp:440-442`, `src/app/play/engine_install.cpp:240-252`, `src/prepare/builder.cpp:125-128` | Les anciens paquets TensorRT ne sont jamais retirés : chaque version de paquet a son dossier `backends/tensorrt/<version>/` (1,3 Go), et `reuse_vendor_file` compte même sur eux pour éviter un téléchargement. Les timing caches `aji-timing-<jeton>.cache` d'un ancien jeton restent aussi. A33-04 traite les moteurs ; ceci est le reste. | Trois montées de TensorRT en un an : 4 Go dans `%LOCALAPPDATA%` ou `~/.local/share`, invisibles dans le lecteur. | Après une installation réussie, garder le paquet en service et le précédent (retour arrière), retirer les autres ; retirer les timing caches dont le jeton n'est plus celui de la machine, avec la purge d'A33-04. | Lu. |
| A36-07 | Mineur | `tools/release/windows/ajp.iss:16-50`, `tools/release/linux/LISEZMOI.txt` | Désinstaller laisse plusieurs Go sans le dire. L'installeur ne propose pas de retirer `%LOCALAPPDATA%\ajp\ajp` (moteurs construits, paquet TensorRT, modèles, caches) ni `%APPDATA%\ajp\ajp` (réglages, journaux, minidumps, `updates/previous`, copie entière de l'installation). Sous Linux, ni le `LISEZMOI` ni les notes ne disent quels dossiers retirer, ni de lancer `ajp --unregister`. SEC-26 ne couvre que les fichiers ajoutés dans `{app}`. | Tatoub désinstalle depuis Paramètres > Applications : 2 à 6 Go restent dans `AppData`, et `updates/previous` suit son profil itinérant. | Une page `[Code]` qui demande « Supprimer aussi les réglages et les moteurs téléchargés ? » (`DelTree` des deux dossiers), avec le `--unregister` du renvoi 19→28. Sous Linux, une section « Désinstaller » dans le `LISEZMOI`, ou une commande `ajp --uninstall-data`. | Lu. |
| A36-08 | Mineur | `src/app/play/paths.h:13-36`, `src/app/play/data_dirs.cpp:47-65` | Sous CachyOS, les caches qui se refont seuls (moteurs construits, `plans/`, `shaders.cache`, `modeles/`) vivent dans `~/.local/share` et non dans `$XDG_CACHE_HOME` (`~/.cache`). Les outils de sauvegarde (Déjà Dup, Borg, Pika) excluent `~/.cache` par défaut, pas `~/.local/share`. | Une sauvegarde de `~` copie chaque jour plusieurs Go de moteurs TensorRT qui se reconstruisent seuls. | Sous Linux, `local_data` rend `$XDG_CACHE_HOME/ajp` pour les entrées de `kHeavy` qui sont des caches ; `move_heavy_data` les y déplace une fois. Le paquet TensorRT et les modèles téléchargés peuvent rester dans `~/.local/share`. | Lu. |
| A36-09 | Mineur | `tests/unit/app/settings_test.cpp:620-645`, `tests/unit/app/` | Aucun test de passage entre versions hors du cas « clés inconnues » de `play.toml` : ni valeur inconnue d'une clé connue (A36-01), ni `settings.toml` d'une autre version (A22-07), ni `calibration.toml` d'une autre version (A36-02), ni `update.toml` à champs inconnus (A36-05). | A36-01 et A36-05 passent sans qu'aucun test ne bouge. | Un jeu de fichiers « écrits par une version future » dans `tests/unit/app/data/`, relus puis réécrits par chaque lecteur, avec une assertion « rien de ce qui est inconnu n'est perdu ». | Lu. |

## Défauts connus revérifiés

| ID | Statut | Note |
|---|---|---|
| QUA-05 | Toujours présent, à préciser | Écritures par renommage sans fsync : `net_build.cpp:194-219` (artefacts Vulkan), `span_derive.cpp:27-35` et `:111-138` (paquet FP8/INT8 et son reçu), `shot_cache.cpp:139-145`, `placebo_open.cpp:299-305`, `engine_install.cpp:539-571` (A36-03). Les quatre premiers sont relus avec contrôle (artefact vérifié, signature des plans, `pl_cache_load`) : une coupure ne coûte qu'une reconstruction. Le reçu FP8 ne porte pas le SHA du paquet (`span_int8.cpp:114-125`) : un paquet FP8 tronqué avec un reçu « passed » n'est pas reconnu (renvoi 20→02 en attente). Le cas grave est A36-03. |
| QUA-07 | Toujours présent | `main.cpp:526` (`std::ignore`) et `file_sync.cpp:86-91` sans nouvel essai : A20-01 et A26-09. |
| WIN-02 | Toujours présent | `main.cpp:478-480` : `move_heavy_data` avant `play()`, donc avant la fenêtre. Voir A19-01 et A34-12. |
| SEC-16 | Toujours présent | `main.cpp:481` à chaque lancement, `data_dirs.cpp:101-139` ; démontré par l'audit 19. |
| UX-19 | Toujours présent | `settings_engines.cpp:82-84` change le brouillon à chaque frappe, et `player_run.cpp:642-646` écrit `play.toml` au plus une fois par seconde (A22-09). |
| WIN-03 | Toujours présent, plus large | `main.cpp:419-440` et `:422` : journaux (`kKeptLogs = 3`, 32 Mo × 3 parties par lancement, voir A33-06) et minidumps dans le dossier itinérant. S'y ajoutent `updates/` et `updates/previous`, copie entière de l'installation (`updater.cpp:238-240`, `:585-587`), jamais retirée après confirmation de la version (A26-06). |
| Renvoi 26→36 (`main.cpp:398-401`) | Confirmé, plusieurs pertes | Retour arrière fichier par fichier : `play.toml` perd les clés de la version récente (A36-01) ; `update.toml` ses champs (A36-05) ; `settings.toml` est refusé entier et mis de côté si la version récente a monté sa version (`toml_fields.h:106`, `player.cpp:262-267`), ce qui relance l'assistant, qui réécrit `play.toml` et la calibration (A22-07 et A25-01, latent tant que `kSettingsVersion` vaut 1) ; les modèles dérivés sont partagés tels quels (A36-04). Le cache des moteurs est sain : la clé suit le contenu, chaque version garde les siens. À confirmer : si la version récente a remplacé le module TensorRT dans le même `backends/tensorrt/<version>/`, l'ancienne le voit en `AbiMismatch` et le réinstalle, puis la nouvelle de même (allers-retours de paquet). |
| Renvoi 29→36 (`player.cpp:574-581`) | Confirmé, cas d'A08-04 | `save_settings` écrit `settings_to_save(chain.running…)` sans `with_kept_choices` : Ctrl+G, Ctrl+I, Ctrl+0 deviennent permanents. L'écriture est synchrone, avant que l'épisode suivant ne s'ouvre ; si son ouverture échoue, `play.toml` garde quand même la chaîne. `rife_auto` coupé par « Annuler et regarder » n'est pas rendu non plus (`settings.cpp:345-366` n'a pas de `keep` pour lui, A22-05, A18-04). Une seule correction pour les trois : A08-04. |
| Renvoi 05→36 (`identity.cpp:161-164`) | Confirmé, à nuancer | Le jeton `driver` vient de `cudaDriverGetVersion` (`tensorrt_backend.cpp:161`, `:305`) : il ne change qu'avec la version CUDA du pilote (12.8 → 12.9 → 13.0), pas à chaque pilote. Chaque saut reconstruit tous les moteurs et laisse les anciens (A33-04). Le retirer de la clé suppose qu'un moteur refusé au chargement se reconstruise ; or `TrtexecBuilder::rebuild` et `rejected_sha256` n'ont aucun appelant (A05-08) : à garder tant que ce chemin n'existe pas. |
| Renvoi 33→36 (`cache.cpp`, A33-04) | Confirmé, et impossible en l'état | La purge « par jeton de pilote » qu'A33-04 propose n'a rien pour décider : le nom `aji-v3-<sha>.engine` est un condensé opaque (`identity.cpp:177-196`), et ni `.sha256` ni `.build.log` ne notent le jeton. Il faut écrire à la publication (`cache.cpp:130-133`) un `.id` à côté de chaque moteur (jeton runtime avec pilote, appareil, date du dernier chargement), puis purger au lancement, hors lecture et sous le verrou de clé, ce dont le jeton n'est plus celui de la machine. La purge ne doit pas s'appuyer sur « clé inconnue de cette version » : elle effacerait les moteurs de l'autre version à chaque retour arrière. |

## Renvois

| Vers l'audit | Où | Constat |
|---|---|---|
| 07 | `src/app/play/machine_calibration.cpp:240-252` | À traiter avec A36-02 et le renvoi 25→07 (moteur mesuré non noté) : un seul jeton d'identité de la mesure. |
| 02 | `src/app/play/span_int8.cpp:114-125` | Le reçu FP8/INT8 ne porte pas le SHA du paquet dérivé : un paquet abîmé avec un reçu valide n'est pas reconnu. |
| 26 | `src/app/play/updater.cpp:670-679` | `confirm_version` ne retire jamais `updates/previous` : la copie de l'installation reste après confirmation, dans le profil itinérant. |

## Points positifs

- `core::write_atomic` (`file_sync.cpp:63-94`) : temporaire au nom unique, fsync du fichier, renommage, fsync du
  dossier ; la grande majorité des fichiers de réglages et d'état passent par lui.
- Le cache des moteurs TensorRT publie avec fsync, vérifie le SHA-256 à la reprise et prend un verrou par clé.
- `play.toml` : lecture lâche valeur par valeur, clés d'avant (GMFSS, `warp_after_span`, `rtx_hdr`, listes de
  langues) relues et testées, fusion avec un autre lancement (P2.14).
- `resume.tsv` et `auto_memory.toml` bornés et tolérants ; `bench.toml` invalidé par le pilote ; le cache des
  plans borné ; les artefacts Vulkan et le cache de shaders contrôlés à la lecture.

## Questions ouvertes

- A36-02 : la calibration doit-elle se refaire seule après une mise à jour (sur secteur, au lancement suivant), ou
  seulement être proposée ? Option retenue dans la correction : proposée par une notice, refaite seule si le
  moteur a changé.
- A36-07 : effacer les données à la désinstallation par défaut, ou seulement sur demande ? Option retenue : sur
  demande, case décochée.
