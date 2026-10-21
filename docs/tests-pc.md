# Essais à faire sur le PC de Tatoub

Ce que le cloud a intégré sans pouvoir l'essayer : il n'a ni RTX, ni écran QD-OLED, ni CUDA
(construit et vérifié seulement jusqu'à la compilation, sur lavapipe pour Vulkan). Chaque ligne dit
quoi lancer et ce qu'on attend. L'extrait de référence reste Re:Zero S4E18, 16:00 → 18:00.

## Construire

Le flux optique matériel NVIDIA a été retiré (fichiers `nvof_session`, `tensorrt_optical_flow`,
`optical_flow_kernels` supprimés) et `kBackendAbi` passe à 4 : reconfigurer `linux-prefix-cuda` et
`linux-prefix-cuda-release`, puis `ajp-gpu ctest --test-dir build/linux-prefix-cuda -j8`.

## Couche de validation Vulkan dans les tests

Les bancs de `vk_compute`, `vk_backend` et `render` demandent `VK_LAYER_KHRONOS_validation`
(`tests/support/vulkan_validation.h`) : chaque erreur de la couche est notée sous le nom du test et
fait échouer l'exécutable, même quand le test passait. `AJP_TEST_VULKAN_VALIDATION=off` la coupe,
`require` (la CI et le préflight) échoue si elle manque ; par défaut, sans la couche installée
(`vulkan-validation-layers` sur Arch), les tests tournent sans elle avec un avertissement.
Première passe attendue sur la RTX : `AJP_TEST_VULKAN_VALIDATION=require ajp-gpu ctest --test-dir
build/linux-prefix-cuda -R "vk_|render"` ; toute erreur relevée est un vrai défaut à corriger, pas
un réglage à faire taire.

## Garder les poses (DRBA)

- `ajp_backend_tensorrt_tests "warp et mélange"` : le warp à poses gardées doit suivre la
  référence CPU (paire précédente mobile, puis tenue, à t = 0,25, 0,5 et 0,75).
- Lecture ×3 puis ×4 avec « Garder les poses (DRBA) » coché, RIFE 4.25 puis 4.22 lite, sur
  16:00 → 18:00 : un personnage animé en deux images sur un fond qui défile doit rester net et
  tenir sa pose pendant que le fond glisse, au lieu de fondre entre deux poses. Le coût attendu
  pour RIFE : une seule passe au milieu par paire.
- À vérifier aussi : pas de saut à la reprise après une coupe (la première paire après une coupe
  utilise le warp simple).

## DistilDRBA (Fluidité → Moteur de l'interpolation)

- Exporter le graphe dans le dossier de RIFE (une minute sur le CPU) :
  `git clone https://github.com/routineLife1/DistilDRBA` puis
  `python -I tools/drba/export_distil_drba.py DistilDRBA --weights DistilDRBA/weights/v2_lite.pkl
  --out <dossier rife>/distil_drba_v2_lite.onnx`. Le script compare son graphe au forward d'origine.
- Lecture ×2 puis ×3 avec le moteur DistilDRBA, TensorRT puis Vulkan, sur 16:00 → 18:00 : relever
  « DistilDRBA » dans le profil GPU (une passe par image créée), les images sautées, et le journal
  de fin (passes sans l'image d'un côté : début, coupes).
- À l'œil, contre RIFE 4.25 avec « Garder les poses » : personnages animés en deux sur un fond
  qui défile ; la dernière image du fichier n'est pas montrée (une image de retard).

## GMFSS Fortuna (Fluidité → Moteur de l'interpolation)

- Exporter les deux graphes (en tête de `tools/gmfss/export_gmfss.py`) dans le dossier de RIFE ;
  le script vérifie que le réseau coupé rend ce que rend GMFSS. Le menu doit cesser de dire
  « Graphes introuvables ».
- `ajp_backend_tensorrt_tests "[gmfss]"` : l'entrée de la paire et le softmax splatting CUDA
  contre la référence CPU (la même que sur lavapipe).
- La première lecture construit les deux moteurs (plusieurs minutes, journal « gpu »). Puis
  720p ×1,5, ×2 et ×3 avec `--stats` : relever le coût de la paire et de chaque image
  intermédiaire, les paires en retard, et ce que la chaîne laisse sans interpolation.
- À l'œil, contre RIFE 4.26 sur 16:00 → 18:00 : traits fins, fond découvert, objets rapides.
- Si un moteur est refusé (forme, nom de tenseur), le lecteur repasse sur RIFE avec la note
  « GMFSS indisponible : RIFE » : le journal dit pourquoi.

## Versions de RIFE et ×1,5

- Fluidité → Modèle RIFE : 4.26, 4.25, 4.13 et 4.6, changés en pleine lecture (chacun construit
  son moteur la première fois). Comparer coût et rendu sur le même extrait.
- ×1,5 (`--rife=1.5`, ou le menu) : 23,976 → 35,964 im/s, une paire sur deux donne une image à
  t = 2/3, l'autre une à t = 1/3 puis la source. Lecture régulière attendue, sans saccade à
  période de deux images, et pts croissants dans `--stats`.

## Smooth Motion

- RIFE ×3 + Smooth Motion (case de Fluidité, puis relancer) : lecture normale attendue, comme le
  premier essai (4 images sautées sur 1312).
- Sans RIFE : la lecture avait presque calé (16 images neuves en 8 s). Comprendre pourquoi avant de
  retirer l'avertissement du menu ; piste : la cadence à la demande de la session contre la
  métrologie du pilote (essayer avec « Cadence fixe » cochée).

## Écran QD-OLED

- Couleurs étendues (Grade → « Couleurs étendues (large gamut) ») : rouges, verts et cyans saturés plus purs, peau
  et gris inchangés ; comparer 0, 0,5 (Éclatant) et 1 avec `--snapshot=X.pq.ppm`.
- Limiteur OLED : la plage vient maintenant de la moyenne d'image de l'EDID ; vérifier dans le
  journal les valeurs tirées pour le MAG 272U E16 (80 et 250 nits attendus pour 265 nits).
- « Cadence fixe » (Soins OLED) : plus de scintillement du VRR dans les scènes sombres, sans images
  sautées en plus.
- Barre de lecture : en pause, elle disparaît après 20 s quand « Effacer la barre en pause » est coché.

## Chaîne et moteurs (passe du 20 octobre)

- **Warp après SPAN** (Fluidité → « Warp après SPAN », relance la lecture) : 1080p → 4K, RIFE ×3,
  avec et sans la case. Relever avec `--stats` le coût par image et les paires en retard, et
  comparer à l'œil les images intermédiaires (traits, bords d'objets rapides) : sans la case,
  SPAN nettoie le warp ; avec, il ne passe qu'une fois par image source.
- **SPAN réutilisé sur les dessins tenus** : le journal de fin dit « SPAN : N passes évitées ». Sur
  un anime en deux, N doit approcher la moitié des paires tenues ; aucune image ne doit rester
  figée pendant un fondu lent (au plus deux réutilisations de suite).
- **Graphes CUDA de TensorRT** : même lecture qu'avant, coût CPU du thread gpu plus bas. Si un
  moteur refuse la capture, il repasse en lancements directs sans message ; un écran noir ou une
  image figée serait un bug de capture (désactiver en revenant au commit d'avant pour comparer).
- **Windows** : `pwsh tools/release/package-windows.ps1` depuis l'invite « x64 Native Tools »
  (`$env:AJP_WITH_CUDA = "ON"` pour le paquet TensorRT), puis l'écran de bienvenue
  (`--first-run`) et une lecture avec TensorRT : le ring passe par des HANDLE NT.
- **FlowJaNai** : suivre `tools/nvof_interp/README.md` (outil NVOF, flux d'ATD-12K, entraînement).

## Chaîne et moteurs (seconde passe du 20 octobre)

- **Mesures en avance** : la suivante est ingérée et mesurée avant RIFE et SPAN de l'image en
  cours. Le journal de fin donne « profil hôte · attente des mesures de la paire » : la médiane
  doit tomber près de zéro (elle valait le temps GPU d'une image). Vérifier qu'aucune coupe de
  scène ni dessin tenu n'est manqué après un seek (la suivante d'avant le seek n'est pas reprise).
- **Mouvement maximal** (RIFE mouvement, ×3 et plus, « Warp après SPAN ») : SPAN de b est enfilé
  entre la mesure et sa relecture ; même image qu'avant, moins de temps mort GPU.
- **Entrée de RIFE** : à ×3 et plus, les images de la paire sont écrites une fois, puis l'instant
  seul. Image identique à avant (comparer une capture à ×4 avec le commit d'avant).
- **Conv directe à plusieurs pixels** (Vulkan sans matrices coopératives : AMD RX 5000/6000,
  Intel, lavapipe) : l'autotuner remesure chaque couche une fois (nouvelle version des candidats).
  `ajp-gpu ctest --test-dir build/linux-prefix-cuda -R "Réseau : Conv"` d'abord (toutes les
  variantes contre l'interpréteur), puis comparer le temps de SPAN avant et après ; le journal
  dit combien de couches ont été mesurées.
- **À essayer, pas fait** :
  - TensorRT : `--tilingOptimizationLevel=3` dans les réglages du moteur, à comparer
    (construction plus longue) ;
  - Vulkan : file de calcul en priorité globale haute, à comparer quand un jeu ou le navigateur
    charge le GPU (risque : l'affichage passe derrière) ;
  - ring en RGB10A2 au lieu de RGBA16 : moitié moins d'octets par image à 4K et haute fréquence,
    à vérifier sur les dégradés du grade HDR.

## Voies de calcul et export « débit » (21 octobre, compilé dans le cloud, jamais lancé sur une RTX)

- **Trois voies** : le journal d'ouverture dit « TensorRT … 3 voies de calcul » (une seule sous
  7,5 Gio, la 4050 mobile). Lecture ×4 avec RIFE puis GMFSS : aucune image intermédiaire cassée
  ou mêlée à la suivante (l'ordre entre voies est automatique, une erreur se verrait là), i/s
  contre le banc de 45,9. `nsys profile --trace=cuda` sur 10 s : SPAN d'une image (voie
  d'agrandissement) doit recouvrir RIFE ou GMFSS de la paire (voie d'interpolation).
- **Doubles tampons et instant de RIFE** : l'entrée de RIFE et l'image intermédiaire alternent
  entre deux tampons, et l'instant suivant d'une paire ne réécrit plus que le plan t
  (`cuMemsetD16Async`, sans noyau). À ×3 et plus, une capture à t = 1/3 identique à celle de
  fd3af4a, avec RIFE 4.25 (11 plans) et 4.22 lite (7 plans).
- **Export « débit »** (TensorRT, plusieurs voies) : le journal dit « export : profil débit : 8
  slots, 10 tampons de conversion, file d'écriture de 8 ». Débit contre 59 et 105 i/s, mémoire
  GPU de l'encodeur relevée avec `nvidia-smi` (le pool CUDA de NVENC grandit avec la file).
- **Écarté : un graphe CUDA par instant** (étude 2, 3.5). La capture figerait les événements des
  voies (anneau de 64 par voie, ordre automatique entre voies) et ceux du profil par étape ; une
  relecture désordonnerait les attentes entre voies. Le gain visé, trois lancements de moins par
  instant, est petit devant les ~7 ms de RIFE, et trtexec ne montrait aucun gain avec
  `--useCudaGraph` (reports/optimisation.md). Les moteurs gardent leur propre graphe. À
  reprendre seulement si `nsys` montre le CPU de lancement comme goulot.
- **Sans objet : double tampon des sommes du splat** : elles servent de brouillon dans un seul
  appel de `soft_splat_all`, sur une seule voie ; rien d'autre ne les lit.

## Linux sur RX 6500M (RDNA2 Navi 24, 4 Go)

Backend Vulkan seul (pas de matrices coopératives, ROCm ne sert pas Navi 24 : MIGraphX n'est plus
proposé à l'écran de bienvenue). Estimation, non mesurée : SPAN Performance coûte environ
173 GFLOP par image à 720p, soit une vingtaine d'images par seconde sur la 6500M ; 540p tient le
temps réel.

- **Réglages de départ** : profil 7 « Léger 1080p » (540p, RIFE ×2, SPAN Performance), puis
  720p si `--stats` laisse de la marge. Garder « Warp après SPAN » et la réutilisation de SPAN.
  RIFE ×3 et plus seulement en RIFE mouvement.
- **Décodage VA-API** : le journal d'ouverture doit dire `vaapi`, plus `logiciel`. Sur un
  portable hybride, VA-API ouvre `renderD128`, souvent le GPU intégré : il décode, l'image passe
  en mémoire hôte puis sur la 6500M (quelques Mo par image, sans souci même en PCIe x4). La 6500M
  n'a pas de décodeur AV1 ; un iGPU Ryzen 6000 et plus en a un. `vainfo` liste ce que chaque
  appareil décode.
- **GPU choisi** : libplacebo prend le GPU dédié avant l'intégré. Le journal donne « backend
  Vulkan : AMD Radeon RX 6500M (RADV NAVI24) ». Sinon, `MESA_VK_DEVICE_SELECT=1002:743f`
  (`MESA_VK_DEVICE_SELECT=list` affiche les identifiants).
- **Sous-groupes 32/64** : l'autotuner essaie aussi l'autre taille de vague de RDNA (nouvelle
  version des candidats, donc tout est remesuré une fois). Avant, `ctest --test-dir
  build/linux-prefix -R "Réseau : Conv"`, puis comparer le temps de SPAN avant et après.
- **VRAM** : surveiller l'utilisation (`radeontop` ou `amdgpu_top`) avec un navigateur ouvert ;
  au-delà de 4 Go le pilote déborde en mémoire système et le débit chute sans erreur.
- **Pilote** : Mesa 24 ou plus récent (RADV), `vulkaninfo --summary` doit montrer
  `VK_EXT_subgroup_size_control` et `shaderFloat16`.

## Linux sur i7-12700, UHD 770 et RX 6500 (bureau)

Mêmes réglages que la 6500M ci-dessus ; la RX 6500 de bureau tourne plus vite (horloge plus haute,
pas de limite de portable), 720p devient le point de départ probable.

- **Écran sur la RX 6500**, pas sur la carte mère : le rendu et la présentation restent sur le
  même GPU. L'UHD 770 sert alors au décodage seul.
- **iGPU activé dans le BIOS** (souvent « iGPU Multi-Monitor ») et pilote `intel-media-driver`
  (iHD) installé : le lecteur décode par VA-API sur l'UHD 770 en priorité (AV1, HEVC 10 bits,
  VP9), puis sur la RX 6500 si le pilote Intel ne gère pas le codec, puis par le CPU. Le journal
  d'ouverture doit dire `vaapi` ; `intel_gpu_top` montre l'UHD 770 au travail. `vainfo --display drm --device
  /dev/dri/renderD128` (puis 129) liste ce que chaque GPU décode.
- **Resizable BAR** activé dans le BIOS : l'image décodée est copiée directement dans la VRAM de
  la RX 6500 (le lien PCIe x4 n'est alors traversé qu'une fois par image).
- **Paquets à installer** : `libva2` et `mesa-vulkan-drivers` (RADV) ; ils viennent du système,
  la release ne les livre pas.

## Ryzen 7 9800X3D et RX 9070 XT (RDNA4, 16 Go)

Profil 1 (1440p, RIFE ×3) au premier lancement. Sous Windows, moteur Vulkan seul ; sous Linux,
Vulkan ou MIGraphX (ROCm 7.2 et plus).

- **Unités matricielles (WMMA)** : le journal doit dire « famille AMD RDNA4, unités matricielles
  utilisées ». La variante coopmat tourne en wave32 (32 exigé par pipeline, comme llama.cpp) ;
  chaque sortie est comparée à celle du noyau direct pendant l'autotune, et « N variantes
  écartées (sortie discordante) » dans le journal de construction dit si le pilote se trompe.
  Plus de zéro écarté : noter le pilote (Adrenalin ou Mesa) et sa version.
- **Autotune** : nouvelle version des candidats (c4), une seule fois par modèle et résolution.
- **Mémoire** : le budget du moteur Vulkan suit la carte (11,5 Gio sur 16 Go, contre 2 Gio
  avant) ; un profil 4K ou GMFSS ne doit plus refuser « budget mémoire atteint ».
- **MIGraphX (Linux)** : le journal doit dire « mémoire partagée avec la chaîne Vulkan ». Les
  tenseurs passent de Vulkan à HIP sans copie par l'hôte ; « tenseurs par l'hôte » signale une
  extension VK_KHR_external_memory_fd absente. Les programmes se recompilent une fois (clé
  « interop »). Comparer le temps par image avec le moteur Vulkan seul : le plus rapide reste.
- **RADV** nomme la carte « AMD Radeon RX 9070 XT (RADV GFX1201) » : MIGraphX doit être proposé
  au premier lancement (la détection ne connaissait que les noms NAVI).

## RTX 4050 mobile (Ada, 6 Go)

Profil 7 (1080p, RIFE ×2) au premier lancement, TensorRT.

- **GPU dédié** : `ajp.exe` exporte `NvOptimusEnablement` et `AmdPowerXpressRequestHighPerformance`.
  Le journal doit nommer la RTX 4050, pas l'iGPU, même sans réglage dans le panneau NVIDIA.
- **Activations partagées** : tous les moteurs TensorRT (RIFE, SPAN, GMFSS) utilisent une seule
  zone de la taille du plus gros. Vérifier la VRAM avec `nvidia-smi` avant et après ; la lecture
  doit rester identique (les graphes CUDA sont recapturés quand la zone grandit).
- **Secteur et batterie** : mesurer sur secteur ; sur batterie le GPU est bridé et le profil 9
  (source) peut être nécessaire.
