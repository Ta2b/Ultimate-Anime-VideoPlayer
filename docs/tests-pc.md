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
- **Registres des noyaux** (étude 3, 4.4 et 4.5) : blocs de 256 fils, au moins 4 blocs par SM
  (64 registres), 3 pour `warp_blend` (80 registres) ; ptxas affiche les registres de chaque
  noyau au build, et un débordement en mémoire locale arrête le build. Vérifié dans le cloud
  (CUDA 13) sur sm_75, 86, 89 et 120 : le build Windows de la release doit passer pareil avec le
  CUDA du PC. sm_120 débordait déjà sans borne (ingest : 255 registres et 712 octets, boucles des
  32 prises déroulées) ; boucles de l'ingest non déroulées, adresses avancées dans `pair_diff` et
  le splat. Le splat lit chaque plan de source une fois pour les quatre coins (quatre fois avant).
- **À mesurer** : `ncu --set full` sur `pair_diff`, `flow_max` (réductions par mélange de
  registres, sans arbre en mémoire partagée), `splat_all` et `ingest`, contre 092b364 ; une capture
  GMFSS ×2 égale à 092b364 à l'ordre des additions atomiques près (déjà libre avant), et les mêmes
  coupes et dessins tenus sur l'extrait de référence (la somme de `pair_diff` change d'ordre).
- **Ingestion en deux passes** (étude 3, 4.1) : les réductions à filtre étiré (triangle, cubiques,
  Spline36) filtrent les lignes (conversion comprise, FP32) puis les colonnes, nx + ny prises par
  pixel au lieu de nx × ny ; la conversion à la taille de l'image a son propre noyau (40
  registres au lieu de 64), Area aussi. Tests `[backend_tensorrt][cuda]` de réduction contre la
  référence, puis le banc `[.bench]` « coût des réductions 1080p vers 720p » contre 015aab1, et
  une source 4K vers 1080p en Spline36 (le cas qui y gagne le plus : 12 prises par axe).
- **Maillage de RIFE écrit une fois** (étude 3, 4.4 ; ABI des backends 7) : chaque tampon d'entrée
  reçoit un `rife_input` complet à sa première paire, puis `rife_pair` (les images et l'instant,
  4 plans de 11 en moins). Capture ×3 RIFE 4.25 égale à 015aab1 après plusieurs paires ; avec
  4.22 lite (7 plans) rien ne change. Un backend d'avant (ABI 6) est refusé au chargement.
- **Warp et composition par objets texture** (étude 3, 4.2) : quand les largeurs des tenseurs
  sont des multiples de 16 pixels (pas de 32 octets de `texturePitchAlignment`) et les adresses
  alignées, `warp_blend`, `warp_blend_poses` et `gmfss_compose` lisent par le filtre bilinéaire
  matériel (objets gardés par le backend, un par adresse et forme) et écrivent deux pixels par
  fil en `half2` ; sinon la lecture exacte d'avant. Tests `[backend_tensorrt][cuda]` « warp par
  objets texture » et « composition par objets texture » (tolérance `kWarpFiltered`, 4/1024 :
  l'émulation de la fraction 8 bits donne 2,1/1024 au pire) ; les anciens tests gardent la voie
  exacte (largeurs 40 et 60). L'instant réglé des poses reste lu exactement : lu par texture,
  l'écart montait à 10/1024 dans l'émulation. À mesurer : `ncu` sur les deux noyaux contre
  3d159a5, et une capture RIFE ×3 avec garde des poses comparée à 3d159a5 (écart attendu sous
  2/1024). Un journal « objet texture refusé » signale un pilote qui refuse le format half.
- **Splat en tuiles** (étude 3, 4.3 ; ABI des backends 8) : chaque tuile 16 × 16 de source
  accumule dans une fenêtre partagée de 32 × 32 cellules placée sur son déplacement moyen (un
  panoramique y reste entier), 4 plans par passe plus les poids, puis une addition globale par
  cellule touchée ; les écarts de plus de 8 px au déplacement moyen vont directement aux sommes
  globales. `exp` une fois par pixel pour tous les plans, `gridDim.z` = travail, normalisation en
  `half2` avec l'instant de GMFSS Union rempli dans le même lancement (`soft_splat_fill`).
  Écart à l'étude : 4 plans par passe et 20 Ko de mémoire partagée (4 blocs par SM) au lieu de 8
  plans et 72 Ko (1 bloc par SM). Tests `[gmfss]` (flots de ±6 px et de −14 à 22 px), puis
  `ncu` sur `splat_all` contre 3d159a5 et une capture GMFSS ×2 égale à l'ordre des additions près.
- **Pas encore : `--kernels=exact`** : la voie exacte n'a pas d'interrupteur ; elle sert quand
  un tenseur n'a pas d'objet texture.
- **Écarté : sous-échantillonner `pair_diff`** (étude 3, 4.4) : la mesure de détection est fixée
  par kernels/spec/detect.h et les tests de référence Vulkan ; la changer change les décisions.
- **Surfaces NVDEC directes** (étude 1, 5.1) : sous TensorRT, NVDEC rend ses surfaces sans la
  copie de sécurité de FFmpeg (`AV_HWACCEL_FLAG_UNSAFE_OUTPUT`) ; le journal d'ouverture dit
  « NVDEC : surfaces rendues sans copie, 18 images tenues au plus » (2 au thread source, 4 à
  l'étage gpu, 12 au backend). Au-delà de 2560 × 1440, la copie reste (« copie de sécurité
  gardée »). Test `[media][decoder][nvidia]` « surfaces directes », puis une lecture longue HEVC
  10 bits sans « données invalides » au journal, et la VRAM comparée à 10b9e9f (environ 80 Mo de
  plus en 1080p).
- **Images intermédiaires écrites dans le slot** (étude 1, 5.2 ; ABI des backends 9) : le warp de
  RIFE mouvement et la composition de GMFSS après SPAN (et le warp sans SPAN) écrivent le slot
  RGBA16 eux-mêmes, sans le tenseur de 50 Mo en 4K ni la passe `write_rgba16` ; le journal dit
  « images intermédiaires écrites directement dans le slot du ring ». Pas avec RTX HDR. Le code
  écrit est celui du chemin par tenseur au bit près (arrondi FP16 puis `unorm16`) : un export
  RIFE mouvement ×2 4K sur 10b9e9f et sur ce commit doit donner des images identiques (md5 par
  image), et `nsys` ne doit plus montrer `write_rgba16` qu'une fois par image source.
- **Cadence fixe à l'export** (étude 1, 5.3) : case « Cadence fixe » (cochée par défaut) dans
  Exporter → Encodage. Aux coupes et aux dessins tenus, la paire n'est pas interpolée et l'image
  précédente est répétée à l'instant manquant : le clip du PC en RIFE ×2 doit donner 529 images au
  lieu de 496, et `ffprobe -show_entries frame=pts_time` des intervalles tous égaux. Le journal de
  fin dit « export écrit : N images dont R répétées ». Décochée, 496 images à cadence variable,
  comme avant.
- **Export par puits CUDA** (étude 1, 5.3 ; ABI des backends 10) : sous TensorRT avec NVENC, à la
  taille de la chaîne (Taille : « sortie de la chaîne »), le journal dit « images écrites en YUV
  dans celles de l'encodeur, sans ring ni libplacebo ». Test `[backend_tensorrt][export]` (NV12 et
  P010, avec et sans tramage, sitings gauche et haut-gauche : un code au plus de la référence).
  Puis un export HEVC 10 bits 4K comparé à un export à une autre taille (même clip, ancien chemin
  libplacebo) : pas de décalage de couleur ni de chroma (`ffmpeg -lavfi psnr` au-dessus de 50 dB
  après mise à la même taille), et `nsys` ne montre plus de rendu libplacebo ni de copie
  `cuMemcpy2D` par image ; la VRAM baisse (plus de ring d'export : environ 500 Mo en 4K au profil
  « débit »). Échap pendant l'export ne doit laisser ni avertissement « écritures YUV jamais
  signalées » ni plantage.
- **H.264 10 bits sous TensorRT** (étude 1, 5.4 ; M1) : test `[backend_tensorrt][host]` (YUV 4:2:0
  planaire 8 et 10 bits, NV12 et P010 en mémoire hôte, cinq images pour faire tourner les trois
  envois). Puis un clip Hi10P, par exemple `ffmpeg -f lavfi -i testsrc2=size=1920x1080:rate=24000/1001
  -t 20 -c:v libx264 -profile:v high10 -pix_fmt yuv420p10le hi10p.mkv` : il s'ouvre (avant, refus
  « aucun des essais »), le journal dit « cuda : format matériel non proposé par get_format » puis
  « décodage logiciel rouvert sur N threads, K paquets rejoués » avec N au-delà de 1, et le profil
  GPU de fin de lecture a une ligne « envoi de l'image décodée » (vers 0,3 ms en 1080p). Lecture
  fluide avec SPAN et RIFE ×2, seek compris.
- **Événements NVDEC réutilisés** (5.4 ; M3) : test `[media][hw][nvidia]` « les événements de fin
  de copie reviennent au pool du flux », puis la suite NVDEC `[media][decoder][nvidia]` inchangée.
- **Rapport des couches TensorRT** (étude 4, 6.1) : après une construction (supprimer un moteur
  pour la relancer), le journal dit « <modèle> <formes> : N couches dont R reformatages ; rapport
  …layers.json » et ce fichier est à côté du moteur. La clé ne change pas : les moteurs déjà
  construits ne sont pas reconstruits. Puis, pour SPAN, RIFE et GMFSS :
  `python3 tools/trt/layer_report.py <moteur>.layers.json --onnx <modèle>.onnx`. SPAN doit sortir
  sans couche FP32 ; RIFE et GMFSS n'en ont que dans leurs îlots (GridSample, Resize, grille).
  Un « ÉCHEC reformatage » (tenseur interne pleine image) est une fusion manquée à corriger dans
  l'export ONNX ; noter le nombre de reformatages par modèle pour la décision NHWC (6.4).
- **Timing cache du GPU** (étude 4, 6.2) : un seul `aji-timing-<16 hex>.cache` dans le dossier des
  moteurs au lieu d'un `<moteur>.timing.cache` par moteur (les anciens peuvent être supprimés).
  Construire SPAN en 1080p, puis changer la hauteur de travail (720p) : la seconde construction doit
  prendre nettement moins de temps que la première (comparer « moteur SPAN prêt en … s » au
  journal). Pendant une construction, ouvrir une deuxième fenêtre du lecteur sur une autre vidéo :
  son OSD dit « En attente d'une autre construction sur ce GPU » jusqu'à la fin de la première, et
  Échap l'annule sans attendre.
- **Écarté : construire deux moteurs en parallèle** (constat P8, même à 12 Go et plus) : deux
  trtexec sur le même GPU chronomètrent leurs tactiques pendant que le GPU alterne entre leurs deux
  contextes, et un moteur mal choisi reste dans le cache. Le timing cache partagé raccourcit les
  constructions suivantes sans ce risque.
- **Lecture des moteurs d'un bloc** (constat T3) : le chargement d'un moteur en cache (journal
  « moteur … prêt en … s ») doit passer sous 0,2 s pour SPAN et rester à peu près proportionnel à la
  taille du fichier pour GMFSS. `IStreamReaderV2` n'est pas utilisé : sans GPUDirect Storage, il ne
  lit pas plus vite qu'un seul `read`.
- **SPAN INT8** (étude 4, 6.3) : `pip install "nvidia-modelopt[onnx]" pillow` dans l'environnement
  Python qui a torch, un corpus d'au moins 100 images SDR d'anime tirées de vos vidéos (commande
  ffmpeg en tête de `tools/span/quantize_span.py`, aucune sortie de RTX HDR ou TrueHDR), puis
  `python -I tools/span/quantize_span.py <SPAN>.onnx --corpus <images> --out "%APPDATA%\ajp\ajp\modeles"`
  (le dossier « modeles » des données du lecteur, à côté de « engines »). Le script dit le PSNR et
  le SSIM de chaque image de validation ; sous 42 dB de moyenne, 38 dB au pire ou 0,985 de SSIM, il
  n'écrit que le reçu. Quand il passe, le lecteur sous TensorRT dit « SPAN INT8 : … (PSNR moyen …) »,
  construit « SPAN INT8 » (environ deux fois plus de couches au rapport, des `Int8` dans les
  formats) et l'OSD montre « SPAN ×2 · <modèle> INT8 ». À mesurer contre le FP16 sur le même clip :
  la ligne SPAN du profil GPU (attendu × 1,3 à 1,6) et l'image côte à côte sur les aplats et les
  traits fins ; sans gain, supprimer les deux fichiers. Le moteur INT8 ne se publie pas.

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
