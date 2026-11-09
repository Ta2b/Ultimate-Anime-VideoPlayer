# Rapport de l'audit 35. Entrées non sûres

Branche : `claude/project-thread-6cgcrt`. Commit audité : `85cbee8`. Date : `2026-11-09`.

## Verdict

Les lecteurs écrits par ajp sont bien bornés et la signature du manifeste ferme le canal de mise à jour, mais la
vraie surface d'attaque sur la cible (un épisode ou des sous-titres téléchargés, analysés par FFmpeg, libass,
FreeType et dav1d livrés dans le paquet) n'a ni veille des failles ni canal pour livrer un correctif (A35-01), et
un MKV forgé à 100 000 chapitres bloque l'ouverture pendant 7 s, 77 s à 300 000 (A35-02, démontré).

## Cadrage retenu

Pas de question posée à Tatoub (consigne du coordinateur) : tout est audité au maximum. Hypothèses retenues :

- La cible est un poste personnel NVIDIA sous Windows 11 ou CachyOS. Un autre programme du même compte n'est pas un
  adversaire (il peut déjà tout faire) : un fichier d'état du dossier de données abîmé compte comme une panne, pas
  comme une attaque.
- L'adversaire réaliste est, dans l'ordre : l'auteur d'une vidéo ou de sous-titres téléchargés (torrent, fansub),
  avec polices jointes ; quelqu'un qui partage un modèle ONNX ou une LUT ; un autre compte du même poste Windows ;
  un dépôt de release ou une clé de signature compromis.
- Qualité d'image : aucune limite d'entrée de cette zone ne dégrade l'image pour gagner du temps. Les deux pertes
  liées aux entrées sont déjà connues : la dernière colonne d'une source impaire (`core/frame.h:140-145`, voulu,
  invisible) et l'atlas des sous-titres en 4K (A17-01). Rien ici ne freine DistilDRBA face à RIFE.

## Modèle de menace

| Entrée | Qui la fournit | Lecteur | Bornes | Fuzz | Risque réel |
|---|---|---|---|---|---|
| Fichier vidéo (conteneur, flux) | n'importe qui (téléchargement) | FFmpeg (`media/demuxer.cpp:160`), NVDEC, dav1d | celles de FFmpeg ; sonde bornée à 20 s hors du thread de la fenêtre (`player.cpp:121-124`) ; aucune borne sur le nombre de chapitres (A35-02) ni sur la taille de l'image (A35-06) | FFmpeg par OSS-Fuzz en amont ; rien sur `describe`, les chapitres, les pistes (A35-08) | **le plus élevé** : parseurs C tiers sur des octets hostiles (A35-01) |
| Polices jointes (MKV, section `[Fonts]`) | idem | libass (fork), FreeType, HarfBuzz | 64 Mio par police (`ass_library.h:22`), aucun total (A17-09) | aucun | élevé : FreeType a eu une faille exploitée en 2025 sur ce chemin (A35-01) |
| Sous-titres (piste, `.ass`, `.srt` à côté) | idem | libass, `subs/srt.cpp`, `text_encoding.cpp` | 64 Mio par script, 1 Mio par paquet, durées bornées (`subtitle_track.h:25-27`, `timebase.h:9-10`) | SRT et encodage, sans indice (A35-08) | moyen |
| Modèles ONNX | l'utilisateur (`--model`), dossiers d'AnimeJaNai | `prepare/onnx_scan.cpp:33`, `gpu/ir`, parseur ONNX de TensorRT, `span_fp8/onnx_weights.cpp` | 2 Gio (`read_onnx_bytes`), profondeur 64, poids externes confinés au dossier (`model_source.cpp:35-43`) | `gpu::ir` seul (SEC-21) | moyen : choix délibéré de l'utilisateur ; A02-06 lit hors bornes |
| LUT `.cube` | l'utilisateur (dossier `luts/`) | `render/cube_lut.cpp` | 16 Mio, 17 à 65 points, nombres finis | oui, avec relecture | faible |
| Réglages (`play.toml`, `user.toml`, `update.toml`, calibration, reprise) | le même compte | toml++ corrigé (`cmake/TomlFixes.cmake`), profondeur 128 | clés bornées une à une | `settings`, `play_settings`, `resume` ; pas `update.toml` | faible (panne, pas attaque) ; A35-07, A35-09 |
| Manifeste de mise à jour | GitHub, signé Ed25519 | `core/update_manifest.cpp` | taille d'archive, https, SHA ; version non assainie (SEC-10) | oui, sans propriété sur la version | faible tant que la clé tient |
| Archive de mise à jour | idem, SHA du manifeste | `tar` du système (`updater.cpp:542-551`) | SHA, sauf SEC-12 | non | faible |
| Index des moteurs (`engines.lock` livré, `engines-nvidia.lock` en ligne) | release GitHub, **non signé** pour le second (A26-02) | `core/model_index.cpp:364` | noms sûrs (`safe_name`), https, SHA, tailles | **aucun** (A35-08) | moyen : il désigne le module TensorRT chargé (A26-02) |
| Fichiers NVIDIA (pypi.nvidia.com) | NVIDIA, SHA de l'index | `core/inflate.cpp`, `engine_install.cpp:263` | taille du `.part` ; décompression sans plafond (SEC-14) | `inflate` | faible |
| Message d'un second lancement | même compte ; sous Windows, tout compte du poste (SEC-15) | `single_instance.cpp:96` | 64 Kio, fichier ordinaire exigé (`home_screen.cpp:218`) | non (trivial) | faible ; moyen sur un poste partagé |
| Fichiers glissés | l'utilisateur | `home_screen.cpp:202` | fichier ordinaire, extension | non | faible |
| Ligne de commande | l'utilisateur | `app/play/options.cpp` | bornes par option | oui | faible |
| Variables d'environnement | le lanceur | `APPIMAGE` (`default_player.cpp:293`, bien échappée), `AJP_PACKAGED` (`updater.cpp:487`), environnement hérité par les outils (`process_posix.cpp:47`) | outils pris dans des dossiers fixes (`engine_tools.cpp:19-36`) | — | faible |

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A35-01 | Majeur | `tools/deps/linux-from-source.sh:35`, `ports/libass/portfile.cmake:9`, `vcpkg.json:5`, `.github/dependabot.yml:5` | Les analyseurs qui reçoivent les octets hostiles sont livrés dans le paquet et figés : FFmpeg `n9.0.2` et libass d'un fork (`the-database/libass@896614fa`) sous Linux, libdav1d de la machine de construction (absente de `system_lib`, `package-linux.sh:48-64`), et sous Windows FFmpeg, libass, FreeType, HarfBuzz et dav1d de la ligne de base vcpkg. Rien ne surveille leurs failles : Dependabot ne suit que les actions GitHub. Et un correctif n'atteindrait personne : aucun manifeste signé n'est publié depuis la 0.5.1 (A28-02). | Une faille de FreeType du type CVE-2025-27363 (police forgée, exploitée en 2025) ou de FFmpeg est publiée. Un MKV de fansub avec une police jointe piégée circule. Sous Windows, la FreeType du paquet est celle de la ligne de base vcpkg : le lecteur reste vulnérable jusqu'à une nouvelle release, que le canal Stable ne propose pas. | Une veille : un job planifié qui compare les versions livrées (FFmpeg, libass et son amont, dav1d, FreeType, HarfBuzz, fribidi, zlib) aux avis de sécurité (OSV) et ouvre une issue. Rebaser le fork libass sur l'amont à chaque correctif. Réparer A28-02 avant tout. Sous Linux, envisager libdav1d du système comme FreeType. | Lu (versions, `dependabot.yml`, `package-linux.sh`) ; le canal muet est démontré par A28-02. |
| A35-02 | Mineur | `src/media/demuxer.cpp:190-199`, `src/app/play/controls_draw.cpp:210-215` | Dédoublonnage des chapitres en O(n²) à chaque ouverture, puis une encoche dessinée par chapitre à chaque image, sans plafond. Le fichier est ouvert jusqu'à quatre fois (`player.cpp:96`, `pipeline.cpp:36`, `shot_probe.cpp:151`, `stills.cpp:276`). | Un MKV de 3,7 Mo à 100 000 chapitres (FFmpeg l'ouvre en 0,13 s) : la boucle d'ajp prend 7,2 s par ouverture. À 300 000 chapitres (11 Mo) : 77 s, la sonde abandonne à 20 s et le fil continue à tourner. Pendant la lecture, 100 000 rectangles ImGui par image. | Plafonner (par exemple 1 000 chapitres gardés, le reste journalisé) et dédoublonner par tri ou ensemble. Ne dessiner qu'une encoche par pixel de la barre. | Démontré : copie exacte de la boucle, `-O2`, 100 000 → 7,15 s, 300 000 → 76,9 s ; fichier produit par `ffmpeg -map_chapters`, relu par `ffprobe` (100 000 chapitres). |
| A35-03 | Mineur | `src/core/update_manifest.cpp:21-28`, `src/app/play/updater.cpp:442` | La préversion accepte tout caractère, retour à la ligne compris, et la version s'affiche telle quelle. Au-delà du chemin (SEC-10), elle sert à composer un message à l'écran. | Manifeste signé avec `version = "1.0.0-a\nVisit evil.example to update"` : accepté, jugé plus récent que la version installée, et le message « … is available » affiche une seconde ligne dictée par le manifeste. Il faut la clé. | La même correction que SEC-10 : préversion limitée à `[0-9A-Za-z.-]` (SemVer), longueur bornée. Ajouter cette propriété à la cible de fuzz `manifest`. | Démontré : `parse_update_manifest` compilé avec toml++ 3.4.0, manifeste accepté, `compare_versions` rend 1. |
| A35-04 | Mineur | `src/core/model_index.cpp:33-40` | `safe_name` accepte les noms réservés de Windows (`CON`, `NUL`, `AUX`, `PRN`, `COM1`…). La version d'un paquet et le nom d'un fichier NVIDIA deviennent un dossier ou un fichier. | `engines-nvidia.lock` (non signé, A26-02) ou un `engines.lock` abîmé avec `version = "CON"` : `backends/tensorrt/CON` désigne la console sous Windows, l'installation échoue avec un message incompréhensible. | Refuser ces noms, avec ou sans extension, sans égard à la casse. | Démontré : `read_engine_index` garde `version = "CON"` et `name = "NUL"`, zéro refus. |
| A35-05 | Mineur | `src/app/play/engine_tools.cpp:79-88` | Les téléchargements des moteurs suivent les redirections (`--location`) sans `--proto =https` ni `--proto-redir =https`, et sans `-q` : `~/.curlrc` s'applique. Le manifeste, lui, impose `--proto =https` (`updater.cpp:90`). | Une redirection vers `http://` est suivie : l'intégrité reste gardée par le SHA, mais le paquet et l'adresse passent en clair, et un `.curlrc` avec `--insecure` ou `--output` s'ajoute aux options. | Ajouter `-q`, `--proto =https`, `--proto-redir =https` aux deux appels de curl. | Lu. |
| A35-06 | Mineur | `src/media/demuxer.cpp:116-117`, `src/app/play/options.cpp:347-351`, `src/app/play/chain_setup.cpp:238` | Aucune borne haute ni de rapport d'aspect sur la taille annoncée par le fichier avant de préparer les moteurs. | Un fichier forgé annonce 16384×4 ou 16384×16384 : SPAN ×2 vise 32768 de large, un moteur TensorRT de forme fixe se construit plusieurs minutes, se garde en cache, puis échoue en mémoire vidéo. Une série de fichiers de tailles différentes remplit le cache des moteurs. | Refuser à l'ouverture une source au-delà de 8192 de côté ou d'un rapport d'aspect de 1:8, avec une phrase claire. | À confirmer : ouvrir un MKV 16384×4 produit par `ffmpeg -f lavfi -i color=s=16384x4` sur la RTX de Tatoub et regarder le journal de construction. |
| A35-07 | Mineur | `src/app/play/updater.cpp:557-564` | `staged_file`, lu dans `update.toml`, n'est pas confiné à `updates/` : sur une empreinte fausse, le fichier désigné est effacé ; sans empreinte, il est extrait sans vérification (SEC-12). | Un `update.toml` abîmé ou réécrit à la main pointe vers une vidéo de l'utilisateur : « Install now » l'efface. | Vérifier que `staged_file` est sous `updates_dir(paths)` (chemins canoniques) avant toute lecture ou suppression ; refuser sinon. | Lu. |
| A35-08 | Mineur | `tests/fuzz/CMakeLists.txt:18-20`, `tests/fuzz/fuzz_parsers.cpp:76-79` | Des lecteurs nourris d'octets extérieurs ne sont pas fuzzés : `read_engine_index` (lu depuis le réseau, non signé), `parse_update_state`, `load_shot_cache`, les poids FP8 (`span_fp8/onnx_weights.cpp`, où A02-06 lit hors bornes), et tout ce qu'ajp fait après FFmpeg (`describe`, chapitres, pièces jointes). La cible `manifest` ne vérifie aucune propriété : elle n'aurait pas trouvé SEC-10. La branche de l'indice d'encodage n'est jamais fuzzée (renvoi 17). | A02-06 a été trouvé à la lecture, pas par le fuzz ; un défaut du même genre dans l'index des moteurs passerait de même. | Ajouter les cibles `engine_index`, `update_state`, `shot_cache`, `onnx_weights`, et une cible `demux` qui ouvre la mémoire par un `AVIOContext` et appelle `Demuxer` ; dans `manifest`, exiger qu'une version lue soit un nom de fichier sûr ; dans `srt`, tirer l'indice du premier octet. | Lu. |
| A35-09 | Mineur | `src/app/play/look_catalog.cpp:42-44` | Le nom d'un look tiré de `play.toml` n'est gardé que contre `/` et `\`. Sous Windows, `C:x.cube` remplace `luts_dir` par le dossier courant du lecteur C:. | Un `play.toml` abîmé fait lire une LUT hors du dossier des looks. Effet limité à la lecture d'un `.cube`. | Exiger `path.filename() == path` et une extension `.cube`, comme pour `image.model` (`settings.cpp:320`). | Lu. |

## Défauts connus revérifiés

| ID | Statut | Note |
|---|---|---|
| MED-14 | Toujours présent | `src/media/av_util.cpp:48-51` : `duration_to_us` rend `INT64_MIN` (donc `kNoPts`) sur une base de temps nulle. `src/core/frame.h:143-145` : `chain_dimension(1)` vaut 0, appliqué à la piste en `demuxer.cpp:116-117`, sans refus d'une piste de taille nulle. A15-10 ajoute le débordement de `pts + duration` (`video_decoder.cpp:167`). Fichier hostile seulement. |
| SEC-10 | Toujours présent, démontré | `update_manifest.cpp:21-28` (`parse_version`) et `:96` : `9.9.9-x/../../../../tmp/evil` est accepté, et `updates/` + version se normalise en `/home/u/.local/tmp/evil` (`updater.cpp:466`). Démontré sur le vrai `parse_update_manifest`. Voir aussi A35-03 (affichage). Demande la clé de signature. |
| SEC-11 | Toujours présent | `updater.cpp:186-188` : `is_relative()` et l'absence de `..` laissent passer `\Users\x` et `D:x` sous Windows. |
| SEC-12 | Toujours présent | `updater.cpp:561` : sans `staged_sha256`, extraction sans vérification ; l'écart entre `verify_file` et `tar` demeure. Aggravé par A35-07. |
| SEC-14 | Toujours présent | `engine_install.cpp:336-346` : le puits de `inflate_raw` n'arrête pas quand `produced` dépasse `file.size` ; la taille n'est comparée qu'à la fin. Le SHA ne protège pas le disque. |
| SEC-15 | Toujours présent | `single_instance.cpp:214-217` : `CreateNamedPipeW(…, nullptr)`, ACL par défaut, nom calculable. `PIPE_REJECT_REMOTE_CLIENTS` écarte le réseau, pas un autre compte local. En plus de ce qu'a vu l'audit 19 : un chemin UNC envoyé par un autre compte est sondé par `is_regular_file` (`home_screen.cpp:218`), donc une connexion SMB part avec les identifiants de la victime (à confirmer sur un poste partagé). |
| SEC-21 | Toujours présent | Graine ONNX de 16 octets (`tests/fuzz/corpus/onnx/seed.bin`), corpus de la nuit dans `work/`, jamais gardé (`fuzz.yml`), `scan_onnx` non fuzzé, CodeQL en `build-mode: none` (`codeql.yml:36`). Élargi par A35-08. |
| Renvoi 26→35 (`update_manifest.cpp:93-96`) | Confirmé | SEC-10 ci-dessus, et A35-03. |
| Renvoi 17→35 (`fuzz_parsers.cpp:40-44`) | Confirmé en partie | iconv est atteint sans indice, par `guessed()` (`text_encoding.cpp:191-195`) ; la branche `hinted()` et tout le chemin Windows (`MultiByteToWideChar`) ne le sont jamais, le fuzz étant POSIX seulement. Dans A35-08. |

## Renvois

| Vers l'audit | Où | Constat |
|---|---|---|
| 33 | cache des moteurs TensorRT (`prepare/builder.cpp:139-145`) | Un moteur par forme, sans purge apparente : des sources de tailles variées font grossir le cache sans fin (à vérifier ; voir A35-06). |
| 17 | `src/subs/subtitle_track.cpp:146-149` | Aucun plafond du nombre d'événements d'une piste ASS intégrée : un MKV forgé de millions de paquets de 1 Mio grossit la mémoire de libass sans limite. |
| 28 | `.github/dependabot.yml` | Seul l'écosystème `github-actions` est suivi : ni vcpkg, ni les dépendances épinglées de `linux-from-source.sh` (support de A35-01). |

## Points positifs

- FFmpeg hérite la liste blanche `file,crypto,data` d'un fichier ouvert localement : un `.mkv` qui contient en
  réalité une description SDP est bien reconnu comme SDP, mais RTP est refusé (« Protocol 'rtp' not on whitelist
  'file,crypto,data' », vérifié avec `ffprobe` 6.1). HLS refuse aussi une liste de lecture dont l'extension n'est
  pas `.m3u8` (`hls.c:3015-3017` de n9.0.2). Une vidéo piégée ne fait donc pas parler le lecteur au réseau.
- Les chemins d'épisodes viennent d'un parcours de dossier : `./nom` ou un chemin absolu, jamais un préfixe de
  protocole FFmpeg (`concat:`, `subfile:`). Les fichiers glissés ou transmis doivent être des fichiers ordinaires.
- Les outils externes (`curl`, `tar`, `nvidia-smi`) viennent de dossiers fixes, jamais du `PATH` ni du dossier
  courant (`engine_tools.cpp:19-36`).
- L'index des moteurs n'accepte que des identifiants sûrs, des adresses https et des SHA-256 ; les fichiers NVIDIA
  doivent venir de `pypi.nvidia.com` et les paquets du préfixe de la release (`engine_online.cpp:56-72`).
- toml++ est corrigé de quatre comportements indéfinis trouvés par le fuzz, avec une profondeur ramenée à 128 ; les
  lecteurs de LUT, SRT, EDID, journaux de construction, reprise et OSD sont fuzzés avec vérification de propriétés.
- Les poids externes d'un ONNX ne sortent pas du dossier du modèle (`model_source.cpp:35-43`), les lectures de
  modèles, scripts, LUT et polices sont bornées en taille.
- FreeType, HarfBuzz, fontconfig et zlib viennent du système sous Linux : leurs correctifs de sécurité arrivent par
  la distribution.
- La fiche du lecteur par défaut échappe correctement `APPIMAGE` dans `Exec=` (`default_player.cpp:53-74`).

## Questions ouvertes

- Faut-il considérer un autre compte du même poste Windows comme un adversaire ? Si oui, SEC-15 monte d'un cran
  (fuite des chemins de vidéos et, à confirmer, des identifiants SMB).
- Le fork `the-database/libass` est-il suivi de près ? Sinon, revenir à libass amont plus le correctif d'AnimeJaNai
  dans `ports/libass` réduirait A35-01.
