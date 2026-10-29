# Audit fonctionnel de bout en bout (branche `audit-fonctions`)

Objectif décidé par Tatoub (12:23) : une 1.0 stable, sans nouvelle fonction, qui marche parfaitement au moins
sur NVIDIA Linux et Windows. Cet audit lance le lecteur pour de bon et regarde ce que chaque fonction active
fait, au lieu de se fier aux tests unitaires.

## Méthode et limites

- Poste : CachyOS, RTX 4070 Ti, écran QD-OLED 165 Hz VRR actif, pilote NVIDIA 615.78. Deux binaires :
  `linux-prefix` (Vulkan seul) et `linux-prefix-cuda` (TensorRT 11.3, RTX HDR/Maxine, NVDEC, NVENC), tous deux
  construits depuis cette branche (code de sortie 0, aucune occurrence de « error » dans le journal).
- Jamais la vraie configuration : `--data-dir=` avec des copies de `settings.toml`, `user.toml`, `play.toml`
  (l'option remplace ces trois fichiers et le journal ; moteurs et modèles restent partagés en lecture).
  `--register` lancé avec `XDG_DATA_HOME` isolé. Toute fenêtre est passée sous `flock gpu-bench.lock`.
- Preuves : journal `ajp.log`, `--stats` (pts de chaque image présentée, d'où les sauts de seek, les pauses et
  les gels), captures `--snapshot` (toasts, sous-titres), `ffprobe` des exports, tests unitaires.
- Fichiers de test fabriqués avec `ffmpeg` dans `/tmp/claude-1000/audit/` (supprimés à la fin) : extrait de
  l'épisode de Re:Zero (chapitres, 12 sous-titres ASS, polices jointes), MKV à deux pistes audio, sans audio,
  sans durée, H.264 brut sans horodatage, une image, 0,5 s, VFR, entrelacé, SAR 32:27, rotation, HDR10, 8K,
  5.1/mono, chemin à espaces, accents et japonais, fichiers vides, tronqués, faux.
- Verdicts : **M** marche, **D** marche avec défaut, **N** ne marche pas, **I** impossible ici (la raison est
  dite). « corrigé » : le verdict est celui d'après le correctif de cette branche.
- Windows : jamais exécuté ici. Tout ce qui le concerne est **non vérifié**, relu seulement (section dédiée).

## A. Lecture

| # | Fonction | V | Preuve |
|---|---|---|---|
| A1 | Lecture / pause (Espace, clic sur l'image, bouton) | M | journal « action : pause / lecture » ; `--stats` : pts figé à 1,835 s |
| A2 | Seek par flèches (±5 s) | M | `--stats` : sauts de +4,96 s et −5,00 s |
| A3 | Seek par clic sur la barre | M | « action : seek 00:00:22.951 » après un clic à mi-barre |
| A4 | Seek par chapitre (Page↑/↓, boutons ⏮ ⏭) | M | 6,59 s → 0,000 ; 1,9 s → 9,968 (début du chapitre suivant) ; 32,9 s → 41,79 |
| A5 | Volume (↑ ↓) | M | toast « Volume x % » ; valeur gardée dans `play.toml`. L'effet audible n'a pas été écouté |
| A6 | Muet (M, bouton) | M | toast « Son coupé » |
| A7 | Vitesse de lecture | I | n'existe pas : seul un calage interne (synchro sur l'écran) change la vitesse |
| A8 | Image par image avant (`.`) | M | `--stats` : pause puis pas de 41,7 ms |
| A9 | Image par image arrière (`,`) | I | `--keys` sépare ses touches par une virgule ; même code (`StepFrame`), `session_test` |
| A10 | Plein écran (bouton, F) | M | clic bouton : la fenêtre passe à 1920 de large puis revient ; même action que F |
| A11 | Double-clic plein écran | I | `--keys` ne fait que des clics simples ; `controls_test` couvre |
| A12 | Piste audio (A, bouton) | M | toast « Audio : Français · VF » ; journal : piste 44,1 kHz mono ouverte |
| A13 | Sous-titres : cycle (J, bouton), choix par langues | M | pistes russe et arabe affichées ; français choisi par défaut (préréglage de langues) |
| A14 | Sous-titres ASS avec polices jointes | M | captures du style Crunchyroll, russe, arabe (lettres liées) |
| A15 | CJK, coréen, police absente | M | SRT externe japonais, ASS sans police : repli libass sur Noto, WenQuanYi, DejaVu, tous les alphabets rendus |
| A16 | Taille des sous-titres | M | capture à l'échelle 2,0 |
| A17 | Reprise de lecture (`resume.tsv`) | M | « reprise à 00:00:34.868 » puis position mise à jour à 37,7 s. Seuil de 30 s : un clip plus court ne reprend jamais (voulu) |
| A18 | Épisode suivant (N) et lecture suivante automatique | M | 01 → 02 → 03 ; en fin de vidéo avec `autoplay_next` |
| A19 | Épisode précédent (Maj+N) | I | `--keys` n'envoie pas Maj ; `episodes_test` |
| A20 | Glisser-déposer d'une vidéo | M | « fichier glissé », la vidéo suivante démarre ; un `.srt`, `.txt`, image est ignoré (la lecture continue) |
| A21 | Glisser-déposer d'un fichier vidéo corrompu | D | retour à l'accueil avec la raison, mais la lecture en cours est perdue (voir « Défauts restants ») |
| A22 | Ouverture par la ligne de commande | M | tous les essais |
| A23 | Fichier absent, vide, faux, tronqué | M (corrigé) | avant : sortie code 1 sans fenêtre (invisible depuis « Ouvrir avec », sans terminal) ; maintenant l'accueil s'ouvre avec la raison, en français |
| A24 | Écran d'accueil : bouton « Ouvrir une vidéo » | I | dialogue de fichiers natif (portail), non pilotable par `--keys` |
| A25 | Écran d'accueil : fichier déposé | M | « accueil : …court.mkv » puis lecture, retour à l'accueil à la fin |
| A26 | Fin de vidéo | M | fichier donné : le lecteur se ferme ; choisi à l'accueil : retour à l'accueil |
| A27 | Vidéo sans audio | M | « audio : aucun, horloge hôte » |
| A28 | Machine sans serveur audio | M (corrigé) | `SDL_AUDIODRIVER=bogus` : avant « SDL_Init : Audio target not available » et rien ne démarrait ; maintenant la vidéo joue sans son |
| A29 | Fichier audio seul (FLAC stéréo/mono, MKA 5.1) | N | refusé : « aucune piste vidéo ». Le lecteur est un lecteur vidéo ; rien ajouté (pas de nouvelle fonction) |
| A30 | Audio 5.1, 7.1, mono dans une vidéo | M | pistes ouvertes pour les trois écoutes ; sur cette machine l'appareil par défaut est multicanal, donc 6 canaux restent 6 canaux (voulu, `audio-spatialisation.md`) |
| A31 | Une image, 0,5 s, VFR, entrelacé, SAR 32:27, HDR10 10 bits | M | tous lus jusqu'au bout, sans avertissement |
| A32 | 8K (7680×4320 H.264) | M | NVDEC refuse, repli logiciel signalé (« décodage logiciel ») ; lecture avec et sans traitement |
| A33 | Fichier sans durée, sans horodatage (H.264 brut) | D | lus jusqu'au bout ; sur le H.264 brut un seek répond « SeekFailed : Operation not permitted » dans l'OSD (message technique) |
| A34 | Chemin avec espaces, accents, japonais | M | `dossier é à/日本語 テスト/アニメ 第1話 é.mkv` |
| A35 | Rotation (métadonnée d'orientation) | N | ignorée : aucune lecture de la matrice d'affichage dans le code. Sans effet sur les épisodes d'anime ; signalé, pas ajouté |

## B. Image

| # | Fonction | V | Preuve |
|---|---|---|---|
| B1 | SPAN ×2 (TensorRT et Vulkan) | M | « moteur SPAN prêt », sortie 1920×1080 |
| B2 | Sans agrandissement (`--no-upscale`) | M | « sans agrandissement » |
| B3 | Résolution de travail (540, 720, `--work-height`) | M | « Mitchell 960x540 », « Area 1280x720 → 2560x1440 ». `--work-height=5` est accepté et joue en 10×5 (CLI seulement, voir restant) |
| B4 | Modèle SPAN : Performance, Balanced | M | Balanced construit en 56 s sous TensorRT, chaîne 2560×1440 |
| B5 | Algorithme de réduction | M | Area et Mitchell vus dans la chaîne |
| B6 | Ordre warp / SPAN (Auto, Avant, Après) | M | les trois lancés ; effet visuel non mesuré |
| B7 | RIFE fixe ×1,5, ×2, ×4, ×8 (Vulkan) | M | ×8 : 73 images sautées, la limite du GPU (normal) |
| B8 | RIFE ×3 fusionné, TensorRT | M | « RIFE fusionné : six moteurs » ; 499 images neuves en 14 s, médiane 20,7 ms |
| B9 | DistilDRBA fusionné | M | « quatre moteurs et noyaux de colle » |
| B10 | GMFSS (paire + fusion) | M | paire construite en 21,7 s, chaîne ouverte |
| B11 | GMFSS union, Apple | I | non lancés (Apple : macOS seulement) |
| B12 | Auto (recommandé) : paliers, plafond | M | ×2 → ×1,5 → ×1,12 → ×1,25 selon la charge ; plafond 60 i/s → « plafond écran ×2,5 » |
| B13 | Auto libre (VRR) | D | s'arme (« Auto libre · 6 i/s »), puis « trop d'images perdues … retour à Auto (recommandé) » en 3,5 s sur cette machine : le repli est voulu, mais l'Auto libre ne tient pas ici (domaine de la PR #130) |
| B14 | Garder les poses | M | basculé sans erreur ; effet non mesuré |
| B15 | Coupes de scène, dessins tenus | M | « 4 coupes de scène, 83 dessins tenus » |
| B16 | HDR : RTX HDR | M | « Maxine 1.3.0.0 · pic 1002 nits », options `--truehdr-*` prises (pic 600, contraste 120…) |
| B17 | HDR adaptatif (Vulkan) ; repli de RTX HDR sans TensorRT | M | « RTX HDR : le moteur vulkan ne l'ouvre pas ; HDR adaptatif à la place » |
| B18 | SDR (`--sdr`), source HDR10 | M | « sortie SDR : ni RTX HDR ni HDR adaptatif, image SDR gradée » ; source 10 bits gardée en HDR10 sinon |
| B19 | Grade, OLED vrai noir, cadence fixe | M | Ctrl+G ×2 sans erreur ; « vrai noir : … pied jusqu'à 0.0100 nit » |
| B20 | Smooth Motion du pilote | M | « couche du pilote NVIDIA demandée » ; présentations ÷2 (959 contre 1900) comme attendu |
| B21 | Échelle finale, débandage | M | `ewa_lanczos` et débandage actifs dans toutes les lectures |
| B22 | Statistiques (Tab), OSD, toasts | M | OSD visible sur les captures ; toasts de volume, muet, piste |
| B23 | Profils 1–9 (Ctrl+1…9), Ctrl+0, Ctrl+I | M | profils 7, 9, 1, 2, 4, 5, 8 passés, « réouverture à … » au même endroit |
| B24 | Profil « 4K » (SPAN sur 1080p) sous Vulkan seul | D | le moteur SPAN 1080p se construit pendant plus de 4 minutes (essai arrêté à 5 min) ; sous TensorRT la construction prend ~1 min. Voir « restant » |
| B25 | Seeks en rafale (120 en 10 s), fuite de slots | M | Vulkan et TensorRT : 198 et 200 images jetées au seek, le flux repart, aucun gel |

## C. Audio

| # | Fonction | V | Preuve |
|---|---|---|---|
| C1 | Écoute casque / enceintes / home cinéma | M | « écoute 0 / 1 / 2 » |
| C2 | Mode Fidèle (B) | M | « écoute 1 fidèle » |
| C3 | Son anime Naturel / Anime / Anime fort | M | lus sans erreur |
| C4 | Profil de virtualisation inconnu dans `play.toml` | D | une seule valeur inconnue fait mettre de côté tout le fichier (`play.toml.invalide-…`) et reprendre les défauts : aucune perte (copie gardée) mais sévère ; impossible de la produire depuis le menu |
| C5 | Appareil de sortie absent | M | repli sur l'appareil par défaut, sans bruit |
| C6 | Liste des appareils | I | interface |
| C7 | Délai audio positif | M | départ à 0,75 s |
| C8 | Délai audio négatif | M (corrigé) | avant : image figée 1,7 s au départ, « audio absent : l'horloge de l'écran mène », gel à chaque seek ; maintenant démarre tout de suite à −50, −300, −5000 ms |
| C9 | HRIR (fichier .wav du casque) | I | aucun fichier HeSuVi sous la main. Chemin absent : « modèle paramétrique à la place » (M) |
| C10 | Volume persistant | M | relu au lancement suivant |

## D. Système et ligne de commande

| # | Fonction | V | Preuve |
|---|---|---|---|
| D1 | Mises à jour : vérification, mode, désactivée sans réseau | I | le binaire local n'a pas de clé de signature (« les mises à jour ne peuvent pas être vérifiées ») ; `updater_test`, `ed25519_test`. Relu : signature, SHA-256, annulation à l'arrêt, restauration en cas d'échec |
| D2 | Lecteur par défaut : inscription | M | `ajp --register` écrit un `.desktop` valide et `mimeinfo.cache` ; `--unregister` l'ôte ; deux fois : « n'était pas inscrit » ; `--register` avec un fichier est refusé |
| D3 | Lecteur par défaut : « Définir par défaut » (xdg-mime) | I | modifierait les associations de Tatoub |
| D4 | Moteurs : installer, changer | I | téléchargement réseau ; contrôle d'espace libre relu (`engine_install.cpp`) |
| D5 | Export MKV / MP4 / MOV | M | `ffprobe` : HEVC 10 bits, 12,01 s, 288 images, audio gardé ; sous-titres ASS gardés en MKV, « non copiée : le MP4 ne la porte pas » pour MP4 et MOV |
| D6 | Export avec chaîne complète, TensorRT/NVENC | M | RIFE ×3 + SPAN, 1920×1080 à 71,928 i/s, 862 images (178 répétées voulues), 5,9 s |
| D7 | Export : dossier absent, destination = source | M | refus nets, source intacte |
| D8 | Export annulé (Échap), disque plein | I | `--keys` ne passe pas dans la boucle d'export ; l'écriture se fait dans `.partial` puis renomme (un `/dev/full` a été remplacé par un vrai fichier) ; `export_writer_test` couvre le refus |
| D9 | Profils 1–9 | M | voir B23 |
| D10 | Réinitialiser (profil d'origine) | I | bouton ; `user_overlay_test`, `profiles_test` |
| D11 | Relancer l'assistant | M | `--first-run` ouvre l'écran de bienvenue |
| D12 | Réglages corrompus | M | `play.toml` et `user.toml` illisibles : mis de côté, défauts, lecture normale |
| D13 | Options : `--bench-level`, `--bench-out`, `--rife`, `--model`, `--work-height`, `--no-upscale`, `--sdr`, `--mute`, `--one-lane`, `--stats`, `--snapshot`, `--keys`, `--first-run`, `--register`, `--truehdr-*`, `--data-dir`, `--start`, `--seconds` | M | tous exécutés. `--bench-out` écrit `level`, `presents`, médianes ; options hors plage refusées en les nommant (`--rife=99`, `--window=10x10`, `--keys=3`…) |

## Corrections de cette branche

1. **Délai audio négatif** (`wiring.h/.cpp`, `wiring_test`, `player_test`) : gel de 1,7 s à chaque départ et
   chaque seek dès −300 ms, puis « audio absent ». L'audio entendu avant l'image exige plus d'images décodées en
   attente que la file n'en tient ; l'horloge audio s'arrêtait faute de son et tout attendait. Au-delà de
   −60 ms l'horloge hôte mène d'emblée (le chemin de la synchro sur l'écran, déjà éprouvé). Mesuré avant/après
   avec `--stats` à −300, −1000, −5000 et +300 ms.
2. **Pas de serveur audio** (`main.cpp`) : `SDL_Init(VIDEO | AUDIO)` empêchait de démarrer ; l'audio s'initialise
   à part et la vidéo joue sans son.
3. **Fichier donné illisible** (`player.cpp`) : l'accueil s'ouvre avec la raison au lieu d'un code 1 silencieux.
   Les essais scriptés (`--hidden`, `--keys`, `--snapshot`, `--bench-out`) gardent le code de sortie.
4. **Messages d'ouverture en français** (`av_util.cpp`, `demuxer_test`) : « format non reconnu ou fichier
   abîmé », « fichier introuvable », « accès refusé », « fichier tronqué ou illisible ».
5. **Aide `--help`** : toutes les touches (elle n'en citait que quatre).
6. **Relance Windows après une mise à jour** (`main.cpp`, non vérifié) : `_spawnv` recevait `c_str()` d'un
   temporaire et des arguments non cités (un chemin à espaces arrivait coupé en deux).

## À corriger par l'audit UI / benchmark

- `settings_system.cpp`, `default_player_prompt` : sous Windows `is_default_player()` rend toujours faux, donc la
  fenêtre « Lecteur par défaut » revient à chaque lancement tant que « Ne plus me le demander » n'est pas coché,
  même après « Définir AJP comme lecteur par défaut » réussi. Mettre `default_player_dismissed` quand
  l'action réussit (ou sur « Fermer » après un succès).
- Le message de repli de l'Auto libre (« retour à Auto (recommandé) ») apparaît en 3,5 s sur cette machine avec
  l'écran VRR : à vérifier côté #130 que ce n'est pas un seuil trop sévère.

## Défauts restants

| Gravité | Défaut | Note |
|---|---|---|
| Moyenne | Un fichier glissé corrompu (extension vidéo) ferme la lecture en cours avant de dire qu'il est illisible | `dropped_file` ne filtre que par extension ; il faudrait sonder avant de fermer la chaîne |
| Moyenne | Un seul réglage invalide dans `play.toml` remet tous les réglages à zéro (copie gardée) | voulu par `settings.h` (« jamais corrigée ») ; atteignable seulement en éditant à la main |
| Basse | Audio seul, rotation d'orientation : non pris en charge | pas de nouvelle fonction |
| Basse | `--work-height` accepte 1 à 99 (image de 10×5) | CLI seulement ; le menu propose des valeurs fixes |
| Basse | Le profil « 4K » demande plus de 4 minutes de construction sur Vulkan seul | TensorRT : ~1 min ; coût connu de la première utilisation |
| Basse | Un seek sur un flux brut sans index répond « SeekFailed : Operation not permitted » | message technique, la lecture continue |
| Basse | Fichiers `.old` laissés dans le dossier d'installation Windows par la mise à jour | non vérifié ; ~ la taille des fichiers chargés, un seul exemplaire |
| Basse | `system_tool` cherche `curl`/`tar` dans `C:/Windows/System32` en dur | Windows installé ailleurs que sur C: : mises à jour et installation de moteurs impossibles (« curl introuvable ») |

## Non vérifié (Windows, macOS, matériel absent)

Relu seulement, jamais exécuté :
- Registre « Ouvrir avec » (`default_player.cpp`, HKCU, `SHChangeNotify`) : séparateurs et guillemets relus, corrects.
- Chemins : le manifeste `utf8.manifest` (page de code UTF-8) fait que `path::string()` rend de l'UTF-8 sous Windows
  10 1903 et plus ; au-dessous, un nom japonais lèverait une exception (`resume.cpp`, `episodes.cpp`,
  `external_subtitles.cpp`).
- Installeur Inno Setup (`tools/release/windows/ajp.iss`) : par utilisateur, sans droits ; aucun nettoyage des `.old`.
- Mise à jour : renommage `.old` d'un exécutable chargé, `tar.exe` lisant un `.zip`, `curl.exe` de System32.
- WinML, WASAPI (via SDL), GPU AMD/Intel, Apple (VideoToolbox, Core ML), TensorRT sous Windows.
- Relance après mise à jour (correctif 6) : le code compile sous `#ifdef _WIN32` seulement.

## Fichiers sensibles touchés

`src/app/main.cpp` (démarrage, relance Windows), `src/app/play/wiring.h/.cpp` (qui mène l'horloge),
`src/app/play/player.cpp` (ouverture), `src/media/av_util.cpp` (messages), `src/app/play/options.cpp` (aide
seulement). Aucun fichier de l'UI (`settings_panel*`, `settings_chain_tabs`, `first_run`, `controls*`) ni du
benchmark (`settings_system`, `player_run`) n'a été modifié.


## Corrigé ensuite (29/10)

- **Fenêtre « Lecteur par défaut » sous Windows** : après un « Définir » réussi, le choix est gardé (`default_player.dismissed`) : la fenêtre ne revient plus à chaque lancement (Windows ne dit pas si AJP est déjà le lecteur par défaut).
- **Valeur invalide dans `play.toml`** : seule la clé fautive est écartée (le défaut la remplace, le journal dit laquelle) ; le reste du fichier reste en vigueur. Un TOML illisible reste refusé et mis de côté (`parse_settings_lenient`, test).
- **Fichier glissé qui ne s'ouvre pas** : le lecteur revient à l'accueil avec la raison au lieu de quitter, même lancé avec un fichier.
- **Repli du mode Rapide** : 5 % d'images perdues (au moins 10) sur 5 s, après 5 s de démarrage (avant : 3 %, 6, 3 s, 3 s), pour ne plus replier sur le démarrage ou une scène lourde passagère d'un écran VRR.
