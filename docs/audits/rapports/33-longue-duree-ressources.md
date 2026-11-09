# Rapport de l'audit 33. Longue durée et ressources

Branche : `claude/project-thread-hyo47t`. Commit audité : `bbf0d5b`. Date : `2026-11-09`.

## Verdict

En régime établi, la lecture ne grossit pas : chaque file, chaque historique et chaque fichier de la chaîne est
borné, et l'essai de 50 fichiers enchaînés sous Vulkan logiciel le confirme (mémoire plate après le 15ᵉ fichier, threads et descripteurs constants). Ce qui casse une
soirée est autour : la mémoire vidéo d'une chaîne fermée reste prise (anneau et réserve CUDA), le lecteur empêche
l'écran et la machine de s'endormir même en pause ou à l'accueil, un réveil de veille sous Windows fait courir
l'horloge jusqu'à la fin de l'épisode, et le cache des moteurs n'est jamais purgé.

## Cadrage retenu

Pas de questions à Tatoub (consigne du coordinateur) : tout est audité au maximum. Retenu de la consigne : cible
NVIDIA sous TensorRT, Windows 11 et CachyOS ; RIFE 4.25, RIFE 4.22 lite et DistilDRBA comptent autant ; ce qui
empêche DistilDRBA d'être bien plus rapide que RIFE est cherché ; le VRR est hors sujet ; toute baisse de qualité
d'image faite pour la vitesse est un défaut, sauf mesure qui la justifie. Point imposé : la veille et
l'hibernation (A29-02, A31-12) entrent dans le banc de soirée.

Le fil tourne sans carte NVIDIA. Le lecteur a été construit (`linux-prefix`, Debug) et lancé sous Xvfb avec
lavapipe : la partie CUDA et TensorRT est lue, jamais exécutée. Tout ce qui touche la mémoire vidéo NVIDIA, la
veille et Windows porte « à confirmer » avec la manipulation.

### Ce qui a été mesuré ici

50 clips de 3 s (H.264 640×360, 24 i/s, AAC) dans un dossier, lecture automatique de l'épisode suivant, sans y
toucher : `xvfb-run ajp --software-gpu --mute --no-upscale --rife=0 --no-resume --data-dir=<neuf> "Serie - 01.mkv"`.
Relevé toutes les 2 s dans `/proc/<pid>` : `VmRSS`, `Threads`, nombre d'entrées de `fd`. Les 50 fichiers passent en
196 s, 49 lignes « épisode suivant », aucune ligne d'erreur au journal, sortie propre à la fin du dernier.

| Fichier | 1 | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Mémoire (Mo) | 289 | 309 | 313 | 322 | 317 | 323 | 319 | 323 | 320 | 323 | 323 |
| Threads | 32 | 32 | 32 | 32 | 32 | 32 | 32 | 32 | 32 | 32 | 32 |
| Descripteurs | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 |

La mémoire monte de 33 Mo sur les 15 premiers fichiers (caches de lavapipe et de libplacebo qui se remplissent),
puis oscille entre 317 et 323 Mo jusqu'au 50ᵉ : pas de pente. Threads et descripteurs ne bougent pas. Cet essai
ne couvre que la chaîne commune (source, décodage logiciel, grade, écran, journal) : ni TensorRT, ni SPAN, ni
interpolation, ni audio (`--mute`), ni sous-titres. Le dossier de données finit à 200 Ko.

### Les trois essais demandés par la fiche (à faire sur le poste NVIDIA, binaire `release`)

Un même relevé pour les trois, une ligne par minute dans un fichier : mémoire du processus (Windows : « Octets
privés » et « Plage de travail » ; Linux : `VmRSS` de `/proc/<pid>/status`), mémoire vidéo du processus et de la
carte (`nvidia-smi --query-compute-apps=pid,used_memory --format=csv -l 60` et
`--query-gpu=memory.used,temperature.gpu,clocks_throttle_reasons.active`), nombre de descripteurs (Linux :
`ls /proc/<pid>/fd | wc -l` ; Windows : colonne « Handles » et « Objets GDI »), nombre de threads, taille du
dossier de données (`ajp.log*`, `crashes`, cache des moteurs, cache des plans). Le lecteur n'écrit aucun de ces
chiffres lui-même (A33-08) : le relevé se fait de l'extérieur.

| Essai | Manipulation | À relever | Tenu si |
|---|---|---|---|
| 8 heures | un fichier de 8 h (ou 20 épisodes enchaînés, lecture automatique), 1080p vers 4K, SPAN + chacun des trois modèles d'interpolation, HDR, sous-titres ASS ; une fois par modèle | le relevé ; dans `ajp.log` : lignes « profil GPU », palier de l'Auto, images manquées par heure, « graphes CUDA coupés » (A21-08), dérive audio et vidéo en fin d'essai | mémoire et mémoire vidéo plates après les 10 premières minutes (pente < 1 Mo/h) ; images manquées de la dernière heure égales à celles de la première ; coût par paire inchangé ; aucun graphe coupé |
| 100 réouvertures | pendant la lecture, alterner 100 fois deux réglages qui rouvrent la chaîne (modèle d'interpolation, cran de qualité), une bascule toutes les 20 s ; puis 100 retours à l'accueil et relances du même fichier | le relevé après chaque réouverture ; durée de chaque réouverture (journal) ; nombre de `TensorRtBackend` créés (3 à 7 par réouverture, voir A08-10) | mémoire, mémoire vidéo, descripteurs et threads de la 100ᵉ égaux à ceux de la 5ᵉ à 2 % près ; durée de réouverture stable |
| 50 fichiers | un dossier de 50 épisodes de tailles variées (720p, 1080p, 4:3, HDR), lecture automatique de l'épisode suivant, sans toucher | le relevé à chaque changement de fichier ; nombre de moteurs dans le cache ; mémoire vidéo à l'accueil après le dernier | mêmes critères ; à l'accueil, la mémoire vidéo du processus revient sous 300 Mo (aujourd'hui non : A33-01) |

### Le banc de soirée, avec veille et hibernation

Trois états du lecteur (lecture, pause, accueil), deux sommeils (veille S3 ou S0ix, hibernation S4), deux
systèmes. Sous Windows : `rundll32 powrprof.dll,SetSuspendState 0,1,0` avec `powercfg /hibernate off` pour la
veille, `shutdown /h` pour l'hibernation. Sous CachyOS : `systemctl suspend`, `systemctl hibernate`, une fois avec
`NVreg_PreserveVideoMemoryAllocations=1` et les services `nvidia-suspend`, `nvidia-resume` actifs, une fois sans.
Dormir 10 minutes, réveiller, noter.

| Cas | Ce que le code laisse prévoir | À noter au réveil |
|---|---|---|
| Lecture, Windows, veille ou hibernation | l'horloge hôte a couru pendant le sommeil : toutes les images sont en retard, l'épisode court à sa fin et le suivant démarre (A33-03) ; l'appareil audio et le contexte CUDA peuvent être perdus (A31-01, A31-10) | position avant et après ; fichier joué ; lignes de `ajp.log` des 30 s suivantes ; retour à l'accueil ou non |
| Lecture, CachyOS, veille | `CLOCK_MONOTONIC` s'arrête : pas de saut ; sans la préservation de la mémoire vidéo, CUDA rend une erreur collante, « error ThreadFailed » (A31-01) | idem, plus `dmesg` (Xid) |
| Lecture, CachyOS, hibernation | comme la veille ; la mémoire vidéo n'est restaurée que si le pilote l'a sauvée | idem |
| Pause, les deux systèmes | l'horloge hôte est figée (`wiring.cpp:120-124`) : pas de saut ; reste la perte du GPU, vue à la reprise | la reprise affiche-t-elle une image, au bon endroit, avec le son |
| Accueil, les deux systèmes | aucune chaîne ; le contexte Vulkan de l'écran doit survivre ; la réserve CUDA et l'anneau de la dernière lecture sont encore là (A33-01) | l'accueil se redessine ; la lecture suivante s'ouvre |
| Veille automatique | elle n'arrive jamais tant que le lecteur est ouvert, même en pause ou à l'accueil (A33-02) | `powercfg /requests` (Windows), `systemd-inhibit --list` (CachyOS) |

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A33-01 | Majeur | `src/app/play/placebo_ring.cpp:84`, `src/app/play/placebo_ring.cpp:240-251`, `src/app/play/placebo_screen.cpp:98-100`, `src/app/play/player_run.cpp:743-745`, `src/gpu/cuda/backend/tensorrt_backend.cpp:394-405` | La mémoire vidéo d'une chaîne fermée reste prise. L'anneau (textures RGBA16 à la taille de sortie, 66 Mo pièce en 4K) n'est détruit que par le `create` suivant ou à la fermeture de la fenêtre : `out.flush()` n'est qu'un `pl_gpu_flush`. La réserve CUDA par défaut garde tout son sommet (`ReleaseThreshold` à `UINT64_MAX`, aucun `cudaMemPoolTrimTo` dans `src`) et survit au backend. | Fin de lecture, retour à l'accueil : plusieurs centaines de Mo à quelques Go restent pris toute la nuit. « Refaire le banc » (`player.cpp:560-564`, dont le commentaire dit « sa mémoire vidéo libérée ») mesure dans des processus enfants avec cette mémoire en moins : sur une carte de 8 Go, la calibration peut recommander une chaîne plus basse que ce que la carte tient, donc une qualité moindre sans raison. Même chose pour `trtexec` qui construit les moteurs de la chaîne suivante, puisque `prepare_chain` (`player_run.cpp:115`) passe avant `create_ring` (`player_run.cpp:176`). Si `Pipeline::open` échoue après `create_ring`, l'anneau reste aussi. | Une méthode `release_ring()` de l'écran, appelée à la fin de `run_chain` après `pipeline->reset()` et avant toute calibration ; à la destruction du dernier `TensorRtBackend`, `cudaDeviceSynchronize` puis `cudaMemPoolTrimTo(pool, 0)`. | Lu. Test à écrire (lavapipe) : `PlaceboScreen::create_ring`, fin de chaîne simulée, puis exiger zéro case d'anneau ; il échoue aujourd'hui. À confirmer : `nvidia-smi --query-compute-apps=used_memory` pendant la lecture 4K, puis à l'accueil, puis pendant « Refaire le banc ». |
| A33-02 | Majeur | aucun appel dans `src` (`SDL_EnableScreenSaver`, `SDL_DisableScreenSaver`, `SDL_HINT_VIDEO_ALLOW_SCREENSAVER` absents) ; `src/app/play/placebo_open.cpp` (création de la fenêtre) | Le lecteur interdit l'extinction de l'écran et la veille automatique pendant toute la vie du processus. SDL 3 coupe l'économiseur dès `SDL_Init` (`SDL_video.c:731-733`) : `SetThreadExecutionState(ES_CONTINUOUS \| ES_DISPLAY_REQUIRED)` sous Windows, inhibiteur d'inactivité Wayland et D-Bus sous Linux. Le lecteur ne le rend jamais. | Pause puis on s'endort, ou fin du dernier épisode et retour à l'accueil : l'écran reste allumé sur une image fixe toute la nuit (marquage d'une dalle OLED, alors que les « soins OLED » existent dans le menu), le PC ne se met jamais en veille. C'est aussi ce qui cache A29-02 : la veille automatique n'arrive pas, seule la veille demandée (capot, bouton, menu) est vue. | `SDL_DisableScreenSaver()` seulement quand une vidéo avance ; `SDL_EnableScreenSaver()` en pause, à l'accueil, dans l'assistant, après la fin. Une fonction pure `must_stay_awake(état)` et son test. | Lu dans les sources de SDL 3.4 construites ici. À confirmer : lecteur en pause, `powercfg /requests` montre `ajp.exe` sous DISPLAY ; `systemd-inhibit --list` ou `gdbus` sous CachyOS. |
| A33-03 | Majeur | `src/core/clock.h:15`, `src/app/play/wiring.h:112-114`, `src/app/play/session.cpp:667`, `src/app/play/player_run.cpp:718-726` | Rien ne voit un trou dans le temps. L'horloge hôte est `steady_clock`, ancrée une fois ; elle mène dès que la vitesse de calage sur l'écran diffère de 1 (réglage par défaut), sans audio ou audio perdu. Sous Windows, `steady_clock` (QPC) continue pendant la veille. Précise le mécanisme d'A29-02. | Lecture, veille de 20 minutes, réveil : la position saute de 20 minutes, chaque image est en retard et jetée, le fichier court à sa fin, la lecture automatique lance l'épisode suivant, et la position de reprise de l'épisode en cours est perdue. Sous CachyOS l'horloge s'arrête pendant la veille : pas de saut, mais le défaut reste pour tout gel long du processus (SIGSTOP, débogueur, machine virtuelle). | Dans la boucle de `run_chain` : si deux tours sont séparés de plus de 2 s d'horloge, mettre en pause, ré-ancrer l'horloge à la position d'avant le trou et l'écrire dans la reprise. Avec A29-02 : pause sur `WM_POWERBROADCAST` et `PrepareForSleep`. | Lu. Test à écrire (pur, sans GPU) : une `Session` en lecture reçoit deux instants séparés d'une heure et la position ne doit avancer que d'une image ; il échoue aujourd'hui. À confirmer : cas « Lecture, Windows » du banc de soirée. |
| A33-04 | Majeur | `src/prepare/cache.cpp:81`, `src/prepare/cache.h:36`, `src/prepare/builder.cpp` (aucun `remove` d'un ancien moteur) | Le cache des moteurs TensorRT n'a aucune purge, ni par âge, ni par taille, ni bouton. La clé contient la taille de la source, le runtime et le pilote : chaque taille de vidéo ajoute un jeu de moteurs (plus `.sha256`, `.key-lock`, `.build.log`, couches, cache de temps), chaque mise à jour du pilote NVIDIA ou de TensorRT rend tout le jeu orphelin sans l'effacer. | CachyOS met le pilote à jour plusieurs fois par mois : chaque fois, tous les moteurs sont reconstruits à côté des anciens. Une bibliothèque de tailles variées (720p, 1080p, 4:3, recadrés) multiplie le tout. Le disque se remplit sans que rien ne le dise ; un disque plein fait ensuite échouer les écritures (audit 28). | À l'ouverture, hors lecture : effacer les moteurs dont le jeton de pilote ou de runtime n'est plus celui de la machine, puis borner par taille (les moins récemment lus d'abord, d'après la date du `.sha256`), comme `shot_cache.cpp:87-98` le fait pour les plans ; dire la taille dans Général, avec un bouton « Vider ». | Lu (`grep` de `remove_all`, `purge`, `prune`, `trim` dans `src/prepare` : seuls les dossiers de travail sont effacés). À confirmer : taille du cache après une mise à jour du pilote. |
| A33-05 | Mineur | `src/gpu/cuda/backend/gpu_timing.h:53`, `src/gpu/cuda/backend/gpu_timing.cpp:52,139`, `src/app/play/placebo_screen.cpp:40,184` | Le « profil GPU » de fin de lecture ne garde que les 262 144 premières mesures par étape (1 048 576 pour le grade), puis cesse d'en prendre sans le dire. | À 144 i/s, le résumé couvre la première demi-heure : un banc de soirée lit des centiles qui ignorent la chauffe et la fin de l'essai, là où la dérive se verrait. | Un histogramme à cases fixes ou un échantillon réservoir, et la mention du nombre de mesures ignorées. | Lu. |
| A33-06 | Mineur | `src/app/main.cpp:427-440` | Deux rotations qui ne se connaissent pas : au lancement `ajp.log` devient `ajp.log.1` à `.3` ; pendant la séance, passé 32 Mo, il devient `ajp.1.log` puis `ajp.2.log`. Les parts de séance ne sont ni archivées ni effacées au lancement suivant. | Une soirée bavarde laisse `ajp.1.log` ; des semaines plus tard il est toujours là, à côté de journaux d'autres séances, et un rapport de problème qui joint « les journaux » mélange les séances ou oublie la part où la panne est écrite. La taille reste bornée (192 Mo au pire). | Au lancement, renommer aussi les parts de séance avec leur journal (`ajp.log.1`, `ajp.1.log.1`) ou les effacer ; faire lire `ajp.1.log` par « Signaler un problème » quand `ajp.log` vient de tourner. | Lu. |
| A33-07 | Mineur | `src/app/play/pumped_task.h:58`, `src/app/play/player.cpp:121-125` | Passé le délai de 20 s, le thread qui sonde un fichier est détaché et garde son descripteur tant que la lecture bloquée ne rend pas la main. | NAS muet : chaque nouvelle tentative d'ouverture ajoute un thread et un descripteur bloqués ; une soirée d'essais en laisse des dizaines, jusqu'à la fin du processus. | Borner le nombre de sondes détachées (refuser d'en lancer une nouvelle sur le même montage tant que la précédente n'est pas rendue) et lier la sonde à l'`interrupt_callback` que le démultiplexeur a déjà (`src/media/demuxer.cpp:166`) pour qu'elle rende la main au délai. | Lu. À confirmer : partage SMB coupé, cinq tentatives, compter les threads. |
| A33-08 | Mineur | `src/app/play/gpu_telemetry.cpp`, `src/app/play/chain_stats.cpp`, `.github/workflows/ci.yml` | Aucun essai de durée et aucun relevé de ressources. Le journal ne dit jamais la mémoire du processus, la mémoire vidéo, le nombre de descripteurs ou de threads ; la CI n'a ni boucle de réouverture ni enchaînement de fichiers (les sanitizers tournent sur les tests unitaires seulement). | Une fuite par réouverture ou par fichier ne serait vue que par un utilisateur après des heures, sans chiffre dans son journal. | Une ligne de journal toutes les 5 minutes et à chaque fermeture de chaîne (mémoire, mémoire vidéo par NVML déjà chargé, descripteurs, threads) ; un test de CI sous lavapipe qui enchaîne 20 clips et 20 réouvertures et exige des compteurs stables. | Lu ; l'essai fait à la main ici est la base de ce test. |

Aucune baisse de qualité d'image propre à la durée n'a été trouvée : l'Auto retient un palier bas après 6 s
(`auto_watch.h:19`) mais remonte après 16 paires propres (`core/auto_rate.h:137`), et aucun réglage ne se dégrade
avec le temps. La seule voie par laquelle la durée touche la qualité est A33-01 (calibration faite avec la
mémoire vidéo de la chaîne précédente encore prise). Pour DistilDRBA, rien de propre à la durée : ses tenseurs
sont alloués une fois à l'ouverture (`upscale_drba.cpp:23-58`), pas par paire ; ce qui lui coûte en continu est
déjà classé (A21-06, attente des mesures à chaque paire ; A21-08, graphes coupés pour toujours après trois
échecs cumulés, ce que huit heures rendent probable).

## Défauts connus revérifiés

| ID | Statut | Note |
|---|---|---|
| TRT-06 | Toujours présent, plus large | `src/gpu/cuda/backend/tensorrt_backend.cpp:394-405` : la réserve n'est pas seulement gardée pendant la lecture, elle survit à la chaîne et au backend. Repris dans A33-01. |
| TRT-13 | Toujours présent, réduit | `src/gpu/cuda/backend/tensorrt_backend.cpp:203` : le destructeur sort si `stream_` est nul, et `stream_` n'est posé qu'à la fin de `create_lanes` (`tensorrt_lanes.cpp:50`). Un échec au milieu de la boucle (`tensorrt_lanes.cpp:35-48`) perd les files et événements déjà créés ; les échecs d'après sont nettoyés. Reste Mineur. |
| TRT-18 | Toujours présent | Noms fixes `ajp-bench`, `ajp-compare`, `ajp-calibration` sous le dossier temporaire, effacés par `remove_all` : `src/app/main.cpp:168,185,213,226,268`, `src/app/play/settings_system.cpp:433,441`, `src/app/play/settings_bench_results.cpp:40`, `src/app/play/machine_calibration.cpp:361`, `src/app/play/engine_screen.cpp:125` (A25-11). Deux lecteurs lancés ensemble s'effacent l'un l'autre. |
| Fiche : « rotation du journal seulement au lancement » | Corrigé | `src/app/main.cpp:434-440` : 32 Mo par part, deux parts, pendant la séance. Reste A33-06. |
| A29-02, A31-12 | Toujours présents | Aucune gestion de la veille. Mécanismes précisés par A33-02 et A33-03, cas d'essai dans le banc de soirée. |
| A21-09 | Toujours présent | `src/gpu/lanes.h:113-115` : sans effet mesurable sur une soirée (les adresses de l'anneau et des tenseurs sont fixes par chaîne). |

## Renvois

| Vers l'audit | Où | Constat |
|---|---|---|
| 03, 09 | `src/gpu/cuda/backend/drba_fused_engine.cpp:344` | A03-09 vaut aussi pour DistilDRBA fusionné : au-delà de 16 graphes (8 jeux d'adresses), lancement direct sans notice. |
| 08 | `src/app/play/engine_build.cpp`, `src/app/play/player_engines.cpp` | Un `TensorRtBackend` (runtime TensorRT, trois files CUDA) est créé puis détruit par moteur préparé et par ouverture : 3 à 7 par réouverture (A08-10). À mesurer dans l'essai des 100 réouvertures. |
| 18, 30 | `src/app/play/chain_hdr.cpp:162` | Le thread qui charge les LUT est détaché s'il est occupé à la fermeture (renvoi 18 vers 30 déjà ouvert). |
| 28 | `src/prepare/cache.cpp` | Disque plein par le cache des moteurs (A33-04) : vérifier que l'écriture d'un moteur refuse proprement. |
| 36 | `src/prepare/cache.cpp` | La purge des moteurs orphelins (A33-04) doit se faire avec le changement de pilote (renvoi 05 vers 36). |
| 31 | `src/app/play/player_run.cpp:510` | Après un réveil, la perte du contexte CUDA arrive en même temps que le saut d'horloge (A33-03) : la reprise d'A31-02 doit repartir de la position d'avant le sommeil. |

## Points positifs

- Tout ce qui s'accumule par image est borné : refus et images non écrites de la session, mesures de l'étape
  d'agrandissement, registre des présentations, horloge des rafraîchissements, file audio.
- Les threads sont des `jthread` rejoints, créés par chaîne ou par fichier, jamais par image ; l'essai fait ici
  retrouve le même nombre de threads et de descripteurs d'un fichier à l'autre.
- Les caches annexes ont leur purge : plans (`shot_cache.cpp:87-98`), minidumps (`crash_dump.cpp:24-33`),
  dossiers de travail des constructions ; la mémoire de l'Auto et la reprise sont plafonnées
  (`auto_memory.cpp:63`, `resume.cpp:123`).
- Le journal est borné pendant la séance et vidé toutes les secondes ; le « profil GPU » écrit à cadence fixe ne
  pèse que quelques Mo sur huit heures.
- Les descripteurs exportés de l'anneau vers CUDA sont fermés des deux côtés, sous Linux comme sous Windows.
- `sha256_file_known` ne relit pas un moteur inchangé : une réouverture ne coûte pas un hachage complet.

## Questions ouvertes

- Sous Windows 11, `steady_clock` avance-t-elle bien pendant la veille et l'hibernation avec la MSVC livrée ?
  Le cas « Lecture, Windows » du banc de soirée tranche ; la correction d'A33-03 vaut dans les deux cas.
- Combien de mémoire vidéo la réserve CUDA garde-t-elle après une lecture 4K avec chacun des trois modèles ?
  C'est ce chiffre qui dit si A33-01 change la recommandation de la calibration sur 8 Go.
- En pause longue, faut-il rendre aussi la mémoire vidéo (anneau et réserve) ou seulement laisser l'écran
  s'éteindre ? Option retenue ici : laisser l'écran s'éteindre, garder la chaîne pour une reprise immédiate.
