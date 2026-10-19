# Règles communes des agents (PC de Tatoub)

Ce fichier est lu par chaque agent lancé sur le PC. Il complète docs/ARCHITECTURE.md du dépôt.

## Le projet

Dépôt C++ « Ultimate Anime VideoPlayer » (privé) : https://github.com/Ta2b/ultimate-anime-videoplayer
(remote SSH sur le PC : git@github.com:Ta2b/Ultimate-Anime-VideoPlayer.git).
Lecteur vidéo anime écrit de zéro : upscaling SPAN (AnimeJaNai), interpolation RIFE et au-delà,
grade HDR adaptatif et RTX HDR, tous les GPU, priorité à la RTX 4070 Ti, lecture temps réel
d'abord, export ensuite. Le plan complet est dans `plan.txt` de cette branche agents/briefs
(texte extrait du document de conception, en français) : lis les sections de ton sujet (grep).
Les résumés et questions ouvertes de la vague 1 sont dans `w1-results.txt`.

## Dépôts de référence (lecture seule, ne JAMAIS modifier)

- Moteur actuel : `~/Documents/animejanai-inference-rtx` (attention : travail local non commité
  de Tatoub, n'y touche pas ; pour lire le commit de référence 118c272, utilise
  `git -C ~/Documents/animejanai-inference-rtx show 118c272:src/kernels.cu` ou un clone à part
  de https://github.com/Ta2b/animejanai-inference-rtx dans ~/ajp-refs).
- Lecteur actuel : `~/Documents/mpv-AnimeJaNai-HDR` (shaders .hook, scripts Lua, native/) et son
  installation `~/.local/share/AnimeJaNai-HDR-linux`.
- Fork mpv : https://github.com/the-database/mpv (commit 5afefa7), à cloner dans ~/ajp-refs si
  besoin (video/filter/vf_animejanai.c, video/out/…, player/video.c, player/display_rate.c).
- Gestionnaire C# : https://github.com/Ta2b/animejanaimanager-hdr (branche
  perf-specialized-manager-v2-20261018), à cloner dans ~/ajp-refs si besoin.
- SDK Maxine VFX : /usr/local/VideoFX (TensorRT 10.9 embarqué dans external/tensorrt).

Ils servent de référence fonctionnelle, pas de modèle d'architecture. Quand tu reprends du code
prouvé, note en tête du fichier le dépôt et le chemin d'origine et ajoute un test qui fixe son
comportement.

## Machine

CachyOS (Arch, pacman ; sudo demande un mot de passe : n'installe rien au niveau système),
i7-10700K 8 cœurs / 16 threads, 46 Gio, RTX 4070 Ti (compute 8.9, pilote 615.78.08), CUDA 13.4
dans /opt/cuda, Vulkan NVIDIA seul installé, KDE Plasma sous Wayland.
Outils en espace utilisateur : `export PATH=~/ajp-tools/bin:~/ajp-tools/venv/bin:$PATH`.
Dépendances construites : `export AJP_DEPS_PREFIX=~/ajp-deps`. TensorRT 11.3.0.99 et trtexec :
`~/ajp-sdk/tensorrt`.

Garde-fous obligatoires (le PC a gelé le 19 octobre sous la charge des agents) :
- Toute construction passe par `ajp-build`, qui prend l'un des 4 emplacements de la machine
  (16 tâches de compilation au plus en tout) et ajoute `-j 4` : `ajp-build cmake --build --preset
  linux-prefix`. N'appelle jamais `cmake --build`, `ninja` ou `make` en direct.
- Tout le reste de lourd (ctest, sanitizers, bancs) passe par `ajp-run <commande>` : tranche
  systemd utilisateur `ajp.slice`, mémoire plafonnée à 30 Go (36 Go au maximum), priorité basse.
- Tout ce qui touche la RTX (tests Vulkan ou CUDA sur NVIDIA, mesures de temps) passe par
  `ajp-gpu <commande>` : verrou `~/ajp-reports/gpu-bench.lock`, une commande à la fois. Garde ces
  commandes courtes (filtre ctest -R sur tes tests GPU). Le reste des tests Vulkan tourne sur
  lavapipe : `with-lavapipe <commande>` (Mesa 26.2.4 en espace utilisateur, le Vulkan de la CI).
- Couche de validation Khronos 1.4.363 : `with-validation <commande>` (ajoute
  `VK_VALIDATION_VALIDATE_SYNC=true` pour la synchronisation). N'ajoute aucun ICD ni couche dans
  un chemin de recherche par défaut.
- Sanitizers : sur la RTX, ASan avec `ASAN_OPTIONS=detect_leaks=0` (le pilote NVIDIA fuit en
  interne) ; fuites et TSan via `with-lavapipe` (TSan plante dans le pilote NVIDIA), avec
  `LSAN_OPTIONS=suppressions=~/ajp-ci/lsan.supp` et `TSAN_OPTIONS=suppressions=~/ajp-ci/tsan.supp`
  (FFmpeg et SDL ne sont pas instrumentés).
- Disque limité (environ 55 Go libres) : supprime tes dossiers de build ASan et TSan dès qu'ils ont
  servi, et ton build/ quand ta branche est poussée.
Une fenêtre de test sur le bureau de Tatoub est permise si elle est petite, brève et jamais en
plein écran ; préfère le hors écran quand c'est possible.

Construction :

    cmake --preset linux-prefix
    ajp-build cmake --build --preset linux-prefix
    ajp-run ctest --preset linux-prefix -E '<tests GPU>'   # puis ajp-gpu ctest ... -R '<tests GPU>'

La CI GitHub tourne sur Ubuntu 24.04 avec lavapipe (Vulkan logiciel) et GCC 13 : ton code doit
aussi y passer. GCC 16 et clang 23 sont plus récents que la CI : n'utilise rien que GCC 13 et
clang 20 n'ont pas, et vérifie le format avec clang-format 20 si tu peux (sinon 23, en signalant
les écarts).

## Règles du code (la CI les vérifie)

- C++23 hôte, avec seulement expected, format, print, span, ranges, jthread et initialiseurs
  désignés en plus du C++20 ; interdits : modules C++, generator, stacktrace, flat_map, mdspan,
  move_only_function. Pas de std::print (GCC 13). Les .cu sont en C++20.
- Erreurs : std::expected<T, Refusal> (core/types.h ; nouveaux RefusalCode à la fin de l'enum,
  avec leur nom dans name()). Aucune exception ne traverse un module ni la boucle d'un thread.
  Une valeur hors plage est refusée, jamais corrigée en silence. Chaque code d'erreur CUDA,
  TensorRT, Vulkan, FFmpeg ou SDK devient un Refusal nommé qui cite l'appel.
- Aucun getenv ni variable d'environnement dans src/. Aucun global mutable. Chaque thread
  possède ses données et parle par Inbox.
- Pas d'allocation ni de copie par l'hôte par image sur le chemin de lecture.
- Un fichier vise moins de 500 lignes ; au-delà de 800, la CI échoue. « Code limpide et clair »
  est l'exigence du propriétaire.
- Commentaires et messages en français ; noms de types en anglais.
- Avertissements en erreurs. clang-format -i sur tes fichiers ; clang-tidy -p build/linux-prefix
  propre sur tes .cpp de src/.
- Dépendances tierces : seulement les cibles ajp::dep::* de cmake/Dependencies.cmake, plus CUDA
  et TensorRT pour les backends NVIDIA (optionnels, derrière AJP_WITH_CUDA).
- Tests Catch2 dans tests/unit/<module>/. Aucun fichier binaire n'entre dans le dépôt : les
  données de test se génèrent dans le test. Un test qui exige le GPU NVIDIA porte l'étiquette
  [cuda] ou [nvidia] et se SKIP avec une raison quand il n'y en a pas (la CI n'en a pas).

## Git

- `git fetch origin` puis `git checkout -b <ta branche> origin/<base indiquée>`.
- Commits en anglais, sujet impératif court, corps qui explique le pourquoi, terminés par ces
  deux lignes exactement :

      Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
      Claude-Session: https://claude.ai/code/session_01HskzHcmMh1oNuwF54Lto5e

- À la fin : `git push -u origin <ta branche>`. Ne pousse jamais sur main, dev, m1/*, w1/*,
  w2/* ni agents/* ; n'ouvre pas de PR et ne fusionne rien : le coordinateur intègre.
