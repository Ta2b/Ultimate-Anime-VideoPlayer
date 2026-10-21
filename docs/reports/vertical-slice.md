# Version jouable « ajp play » sur le PC Linux (pc/vertical-slice)

État au 19 octobre 2026, soir. RTX 4070 Ti, pilote 615.78.08, TensorRT 11.3.0.99, KDE Plasma
Wayland, écran MSI MAG 272UP QD-OLED 165 Hz, VRR « Always ».

## Lancer

    cd ~/ajp-work/vertical-slice
    export PATH=~/ajp-tools/bin:~/ajp-tools/venv/bin:$PATH AJP_DEPS_PREFIX=~/ajp-deps
    cmake --preset linux-prefix-cuda-release -DAJP_TENSORRT_ROOT=$HOME/ajp-sdk/tensorrt
    ajp-build cmake --build --preset linux-prefix-cuda-release --target ajp   # lecteur optimisé
    # (linux-prefix-cuda : la version de débogage, pour les tests)
    build/linux-prefix-cuda-release/src/app/ajp play "<épisode.mkv>"            # fenêtre 1280x720
    build/linux-prefix-cuda-release/src/app/ajp play "<épisode.mkv>" --fullscreen
    build/linux-prefix-cuda-release/src/app/ajp play "<épisode.mkv>" --truehdr   # avec RTX HDR

Lanceur : `~/ajp-tools/bin/ajp-anime "<épisode.mkv>"` (plein écran ; la chaîne suit le menu des
paramètres).

Touches : espace pause, flèches seek 5 s, haut/bas volume, m muet, j sous-titres suivants, a piste
audio suivante, p paramètres, Tab statistiques, f plein écran, Échap (ferme le menu, puis quitte
le plein écran), q quitter. Souris : clic pause, double clic plein écran, clic droit pause,
molette volume.
Options : `--rife=N` (0 aucune, défaut 3), `--truehdr` (RTX HDR), `--truehdr-peak=NITS` (défaut :
la crête vue par le contenu sur l'écran, 766 nits ici), `--truehdr-contrast/-saturation/-gray`,
`--work-height=N` (défaut 720), `--model=…`, `--rife-model=…`, `--start=S`, `--seconds=S`,
`--stats=FICHIER` ; `ajp play --help`.

## Chaîne

1920x1080 (NVDEC) → Area 1280x720 (fusionnée à l'ingestion, une passe CUDA) → RIFE 4.26 ×N à
1280x720 (rempli à 1280x768) → SPAN AnimeJaNai V3.1 Sharp1 Performance ×2 → 2560x1440 → [--truehdr : RGBA8 tramé → Maxine
TrueHDR 1.3 → RGB10A2 PQ BT.2020] → ring → libplacebo (sortie HDR10 PQ BT.2020, sous-titres libass
après l'interpolation, OSD) → écran.

- RTX HDR : le SDK Maxine de /usr/local/VideoFX est importé une fois dans le magasin du lecteur
  (~/.local/share/ajp/ajp/sdk, seules les bibliothèques que TrueHDR charge) et revérifié à chaque
  lancement ; pic par défaut = marge HDR de KWin × 203 nits (766 nits sur l'écran du PC).

- Modèles lus en place, jamais copiés : SPAN depuis
  `~/.local/share/AnimeJaNai-HDR-linux/animejanai/onnx/2x_AnimeJaNai_HD_V3.1Sharp1_Performance_SPANF3_b5f48_unshuffle_fp16.onnx`,
  RIFE depuis `…/animejanai/rife/rife_v4.26.onnx` (même fichier que la source du lecteur actuel,
  même clé de moteur aji-v3-a7f6e533…).
- RIFE ne tourne que sur une paire qui bouge : coupe de scène (score > 0,150) et dessin tenu (pire
  bloc 32 × 32 sous 0,70 niveau) laissent l'image seule, l'écran garde la précédente.
- Présentation à la demande : l'écran n'est recomposé que si l'image, les sous-titres, l'OSD ou la
  taille changent ; avec le VRR, il suit le rythme des images.

## Mesures (fichier testé : « [Erai-raws] Tensei Goblin dakedo Shitsumon Aru - 02 [1080p] », ASS fr)

| Essai | Images neuves | Sautées | Intervalle médian (σ) | Paires interpolées |
|---|---|---|---|---|
| SPAN seul, 1080p → 4K, présentation à chaque vsync, 32 s | 753 | 15 (démarrage) | 42,5 ms (2,4) | – |
| Area 720 + SPAN 1440p, présentation à la demande, 20 s | 470 | 0 | 41,8 ms (0,9) | – |
| RIFE ×2, dialogue (5:00), 20 s | 601 | 0 | 40,4 ms | 123 / 485 (359 tenues, 3 coupes) |
| RIFE ×3, dialogue, 20 s | 722 | 0 | 15,0 ms | 123 / 484 |
| RIFE ×4, dialogue, 20 s | 843 | 0 | 10,8 ms | 123 / 484 |
| RIFE ×4, ouverture (0:50), 25 s | 1185 | **304** | 19,8 ms (19,0) | 303 / 587 |
| RIFE ×3, ouverture, 25 s | 1224 | 8 | **14,0 ms (0,6)** | 319 / 603 |

Défaut retenu : ×3 (72 im/s). Les essais avec RTX HDR restent hors du dépôt (licence de NVIDIA).
Sous-titres : 0 présentation sans sous-titres en attente partout.

Temps d'un réseau seul (trtexec, --noDataTransfers, sous le verrou GPU, bureau actif) :
- SPAN Performance 720p → 1440p : médiane 4,37 ms (référence du lecteur actuel : 4,24 ms) ;
- RIFE 4.26 à 1280x768 : min 7,8 ms, médiane 12,7 ms (bureau et navigateur actifs) ;
- RIFE 4.26 à 2560x1472 (l'autre ordre, SPAN puis RIFE à 1440p) : min 72,8 ms, médiane 115 ms.
  Un seul RIFE à 1440p dépasse le budget d'une paire (41,7 ms) : RIFE reste avant SPAN.

## Interface (19 octobre, soir)

Superposition Dear ImGui dessinée par libplacebo par-dessus l'image (renderer repris de
render/imgui_layer), recomposée seulement quand elle change (≤ 60 Hz), police du bureau à
l'échelle de l'écran (×2 ici).

- Barre de lecture (812e305) : progression cliquable et glissable (bulle du temps au survol, seeks
  au point clé pendant le glissement, exact au lâcher), lecture/pause, volume jusqu'à 150 % et
  muet, temps, titre, plein écran ; paraît au mouvement de la souris, s'efface après 2 s (curseur
  compris), reste en pause ou sur un menu. Les sous-titres montent au-dessus de la barre.
  Instantané : ui-bar.png.
- Pistes (c87d21e) : menus sous-titres et audio par langue, changement sans coupure (époque du
  pont audio, SelectTrack + seek pour subs). ui-tracks.png.
- Préparation des moteurs (14d0e4f) : fine barre + « Préparation du moteur SPAN · 39 % », détails
  avec Tab. Pourcentage au temps écoulé sur la durée attendue (journaux du cache), trtexec n'ayant
  pas de repère pendant le choix des tactiques. ui-build.png.
- Menu des paramètres (246abcf) : onglets Image, Fluidité, RTX HDR, Son, Statistiques, Moteurs ;
  play.toml dans ~/.local/share/ajp/ajp/. En direct : RIFE (moteur chargé), réduction, réglages
  RTX HDR, délai audio, statistiques. Après « Appliquer » (la chaîne se rouvre à la même
  position, ~7,5 s) : modèle, hauteur de travail, RTX HDR allumé/éteint. ui-settings.png,
  ui-settings-hdr.png.
- Bug trouvé : libplacebo 7.360, `vk_tex_upload` passe la fin du rectangle comme taille
  (imageExtent = rc.x1) ; une mise à jour de texture hors de l'origine débordait et bloquait le GPU
  (vu sous validation). Contourné en envoyant depuis l'origine.

Coût des réductions fusionnées 1080p → 720p (9b4e5ef, moyenne sur 300 images, verrou GPU) :

| Filtre | Temps par image |
|---|---|
| Area (défaut) | 0,09 ms |
| Bilinear (triangle étiré) | 0,29 ms |
| Catmull-Rom (B 0, C 0,5) | 0,86 ms |
| Bicubic (B 1, C 0, comme libplacebo) | 1,05 ms |
| Mitchell (B = C = 1/3) | 1,06 ms |

- Profils (699c64c) : 9 profils livrés (TOML v5 de core) et user.toml par-dessus, Ctrl+1 à Ctrl+9
  et onglet Profils ; un profil fixe la chaîne (réduction, RIFE, SPAN, RTX HDR), play.toml garde le
  reste. Au premier lancement avec profils, la chaîne de play.toml devient le profil de démarrage.
  ui-profiles.png.
- Écran de bienvenue : au premier lancement (settings.toml absent), ou avec --first-run. GPU lu par
  Vulkan (nom, mémoire, matrices coopératives = cœurs Tensor), moteurs de core::recommend_engines
  avec badges et présélection (Vulkan toujours ; TensorRT et Maxine cochés sur une RTX ; MIGraphX
  et OpenVINO grisés), profil de démarrage conseillé (1 pour une RTX de 11 Go et plus). Commencer
  écrit settings.toml et user.toml. Sans TensorRT : ni SPAN ni RIFE ; sans Maxine : pas de RTX HDR
  (dit au journal). ui-welcome.png.

- Grade HDR « Éclatant » : anime-hdr-grade (9 passes) et la passe du vrai noir repris de pc/render,
  posés par libplacebo sur OUTPUT quand la sortie est en PQ, sur les deux routes : RTX HDR
  (TrueHDR puis grade) et HDR natif (SDR converti en PQ par libplacebo, aspect TrueHDR émulé avec
  les réglages de RTX HDR). Onglet Grade : préréglage Éclatant (défaut) / Naturel (le grade du
  lecteur actuel) / Personnalisé, curseurs Éclats, Aplats lumineux, Couleurs, Noirs profonds,
  Garde-fou des couleurs saturées, gamut P3, tout en direct. Onglet RTX HDR : route RTX HDR / HDR
  natif / SDR. Sans Maxine, ou si RTX HDR ne s'ouvre pas, la route native prend la place (dit à
  l'écran). Garde-fou : épaule douce du canal le plus fort dès 70 % du pic, pondérée par la
  saturation (un orange vif poussé fort reste un dégradé au lieu d'un plateau à 996 nits ; gris et
  blanc inchangés, test GPU). Coût : 0,44 ms de GPU par image présentée en 2560x1440. Aperçus SDR
  (approximation du HDR) : grade-966.jpg, grade-1003.jpg, grade-1047.jpg ; menu : ui-grade.png.
- Vrai noir OLED : détection par l'EDID (/sys/class/drm, sans libdisplay-info) : DisplayID 2.0 bloc
  0x21, puis table des modèles connus (fabricant + product id EDID, ou nom), puis heuristique (noir
  ≤ 0,001 nit ou code ≤ 2, et plein écran ≤ 0,5 × pic). Écran de Tatoub : QD-OLED « penta tandem »,
  EDID « MAG 272U E16 », fabricant CND, product id 0x8CD7 (36055), HDR10, 1037 nits, moyenne 265,
  noir code 2 (0,00064 nit), DisplayID 1.x sans bloc 0x21 → reconnu par la table (« QD-OLED,
  modèle connu ») : « Auto » active le vrai noir (vérifié sur son écran). Une première déduction
  « Rapid IPS » tirée d'une recherche web était fausse. Onglet Grade → Noir absolu : Auto /
  Activé / Désactivé, seuil en pas de code (2 par défaut). Seuils calés en direct : route RTX HDR,
  la rampe de 32 marches passe par TrueHDR sur le thread gpu (au plus 4 fois par seconde) ; sur les
  deux routes, elle passe ensuite par libplacebo et le grade (renderer et grade à part) et se relit
  en nits. Mesuré : TrueHDR met déjà les codes 16 et 17 à 0 nit et le code 18 à 0,0003 nit ; au
  seuil 2 le pied finit à 0,0003 nit, au seuil 8 à 0,0197 nit.
- Vérifié sur une mire (32 bandes 16 → 47, grain 17 ± 1 tiré à chaque image, dégradé 16 → 40) :
  bandes monotones, dégradé monotone et jamais plus clair qu'avant ; bascules allumé ↔ éteint du
  grain d'une image à l'autre : 11,9 % sans vrai noir, 3,0 % au seuil 2, 0 % au seuil 8. Sur
  l'extrait (17:28, 10,8 % de l'image sous 0,5 nit) : seuil 2 identique à sans ; seuil 8, un peu
  plus de bascules au niveau 0,005 nit (2,05 % contre 1,07 %) dans un plan qui bouge.
- `--snapshot-frames=N` : N présentations de suite (nom-001.pq.ppm…).
- Smooth Motion du pilote (615.78.08, NVPRESENT_ENABLE_SMOOTH_MOTION=1, couche implicite
  VK_LAYER_NV_present) : la couche se charge et accepte la swapchain HDR10 (R10G10B10A2 2560x1440,
  « HW Flip Metering »). Avec RIFE ×3 sur 16:00 → 16:20 : lecture normale (1312 images neuves,
  4 sautées, nos présentations à 14,0 ms) ; la cadence des images ajoutées par le pilote ne se voit
  pas d'ici (journal muet là-dessus). Sans RIFE (×1), la lecture a presque calé (16 images neuves en
  8 s contre 192) : à comprendre avant d'en faire une option.
- `--snapshot=X.pq.ppm` garde la sortie HDR telle que l'écran la reçoit (PQ BT.2020 16 bits, grade
  compris) ; `--keys=0:space` fige la première image, identique d'un essai à l'autre.

Une fois par image source (24 par seconde) : au plus 2,5 % du budget. Les cubiques reconvertissent
chacun de leurs 7 × 7 pixels de source ; une version en deux passes est à faire. SSimDownscaler :
à venir.

## Ce qui manque encore

- RIFE ×4 sans images sautées dans les scènes qui bougent : lots SPAN, graphes CUDA, encodeur
  partagé (travail de la parité, M3).
- Dessins tenus : l'interpolation ne s'étale pas encore sur toute la durée d'un dessin (timeline des
  dessins, M4) ; le mouvement se fait entre la dernière copie d'un dessin et le suivant.
- RIFE Fast (mouvement à basse résolution, warp à la sortie) : pas encore.
- Réouverture de la chaîne plus courte : garder le backend et les moteurs chargés quand ni le
  modèle ni la taille ne changent (aujourd'hui ~7,5 s, dont 1,2 s pour un second backend et 2,3 s
  pour la vérification du cache de RIFE).
- Installation des moteurs depuis l'écran de bienvenue (index haché) : les lignes absentes du poste
  restent grisées.
