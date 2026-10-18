export const meta = {
  name: 'pc-phase-2',
  description: 'PC : backend Vulkan, moteur gpu, render, present et export sur la RTX, puis intégration de bout en bout',
  whenToUse: 'Sur le PC de Tatoub : vague 2 du lecteur sur la vraie carte',
  phases: [
    { title: 'Implémenter', detail: 'un agent par module, dans son worktree' },
    { title: 'Relire', detail: 'deux relectures adverses par module' },
    { title: 'Intégrer', detail: 'fusion des branches et test de bout en bout sur la RTX' },
  ],
}

// args.briefs : chemin absolu du checkout de agents/briefs sur le PC (par ex. /home/ta2b/ajp-briefs)
const B = args.briefs
const CONTRACTS = 'src/core/gpu_device.h, src/core/ring.h, src/core/frame.h, src/core/messages.h, src/core/resolve.h (DisplayInfo), src/core/thread.h, src/gpu/backend.h (ABI 2), src/gpu/kernel_ops.h, et la cible ajp::core_log (noms exacts : ceux de origin/w2/fondations)'
const FONDATIONS = `
Les contrats partagés de la vague 2 (${CONTRACTS}) sont écrits en ce moment dans le cloud sur la branche w2/fondations, avec les correctifs transverses de la vague 1 (démarrage des threads, journal, kNoPts…). Elle n'est peut-être pas encore poussée quand tu démarres :
- vérifie avec « git ls-remote origin w2/fondations » ; si elle existe, fusionne-la tout de suite dans ta branche (git fetch origin w2/fondations ; git merge origin/w2/fondations) ;
- sinon, commence par ce qui ne dépend pas des contrats, derrière tes propres interfaces internes, revérifie toutes les quinze minutes environ, et fusionne-la dès qu'elle paraît ; elle peut encore changer après une relecture : refusionne-la avant ton dernier commit ;
- tu ne modifies jamais ces contrats de toi-même : s'il en manque un morceau, ajoute-le dans ton module et décris précisément le besoin dans shared_changes.
`
const COMMON = `
Lis d'abord ${B}/regles.md en entier : il fixe la machine (RTX 4070 Ti, Vulkan NVIDIA seul, CUDA 13.4, Wayland), les dépôts de référence, les règles du code et Git. Puis lis dans le dépôt docs/ARCHITECTURE.md, CMakeLists.txt, cmake/*.cmake, src/core/*.h, src/gpu/backend.h et le CMakeLists.txt du module qui te concerne. Le plan du projet est ${B}/plan.txt : lis les sections de ton module (grep) avant d'écrire du code. Les comptes rendus de la vague 1 sont dans ${B}/w1-results.txt ; le profil de calcul mesuré de la RTX (matrices coopératives, débits) sera dans ${B}/rtx-compute-profile.md s'il est déjà poussé (git -C ${B} pull).
${FONDATIONS}
Le socle de calcul Vulkan (src/gpu/vulkan/compute : contexte, chargement de Vulkan sans liaison, ressources, pipelines SPIR-V embarqués, command buffers rejoués, sémaphores timeline, noyaux GLSL de kernels/spec dans src/kernels/glsl) est sur la branche origin/w2/vk-compute ; ses correctifs NVIDIA arrivent sur origin/pc/vk-compute-nvidia.

Quatre autres agents écrivent en même temps les autres modules de la vague 2 sur les mêmes contrats, sans pouvoir se parler : respecte les contrats à la lettre.

Tes tests tournent ici sur la vraie RTX (Vulkan NVIDIA, VK_KHR_cooperative_matrix et VK_NV_cooperative_matrix2 présents), mais la CI GitHub les fera tourner sur lavapipe (Vulkan logiciel, sans unités matricielles, sans surface Wayland) : chaque test GPU doit passer sur les deux ou se SKIP avec une raison quand une capacité manque, jamais réussir en silence. Garde-les petits. Mesures de temps : sous flock ~/ajp-reports/gpu-bench.lock, avec échauffement et plusieurs passes. Fenêtres : petites et brèves, jamais en plein écran.

Pas d'allocation ni de copie par l'hôte par image sur le chemin de lecture : l'image reste sur le GPU du décodeur à l'écran ; les copies hôte ne sont permises que dans les tests, l'export logiciel et les outils.
`
const MODULES = [
  {
    key: 'vk-backend', branch: 'pc/vk-backend', base: 'w2/vk-compute',
    scope: 'src/gpu/vulkan/ (hors compute/, que tu peux étendre de façon additive), nouvelle bibliothèque partagée ajp_backend_vulkan (add_library MODULE, chargée à l\'exécution, exporte ajp_backend_create et ajp_backend_destroy), et tests/unit/vk_backend/',
    brief: `Notre moteur Vulkan : le backend ajp_backend_vulkan qui implémente gpu/backend.h (ABI 2) et KernelOps sur le socle src/gpu/vulkan/compute, et exécute les réseaux importés par gpu_ir. Lis la section « Notre moteur Vulkan » du plan.
1. Point d'entrée et appareil : ajp_backend_create adopte le GpuDevice du rendu (mêmes VkDevice et VkImage que libplacebo : aucune interop) ou, sans appareil fourni, crée le sien (outils, banc, tests). capabilities() et match() par UUID.
2. KernelOps : branche chaque opération du contrat sur les noyaux du socle (ingestion, rééchantillonnage, padding RIFE, warp et mélange, écriture dans un slot du ring, coupe de scène et différence par bloc avec lecture asynchrone servie à la paire suivante). import() d'une image Vulkan externe (slot du ring, image FFmpeg Vulkan) sans copie, avec attente du sémaphore timeline fourni.
3. Exécution des graphes gpu_ir : build() importe l'ONNX, applique les passes, choisit un noyau par opération et écrit un Artifact sérialisé et versionné (IR optimisée, variantes choisies, constantes de spécialisation, hachage des SPIR-V) dans le dossier de cache ; load() crée les pipelines, alloue l'arène selon le plan mémoire de gpu_ir et pré-enregistre un command buffer par forme d'entrée ; enqueue() le rejoue sur la file compute avec sémaphores timeline. Shaders GLSL génériques (dans src/gpu/vulkan/shaders/ ou src/kernels/glsl/) : Conv 2D NHWC (groupes, dilatation, strides, pads) en FP32 et FP16 empaqueté, avec tuilage réglé par constantes de spécialisation et activation et résiduel fusionnés ; opérations élément par élément ; Concat, Slice et Split comme vues quand c'est possible ; DepthToSpace, SpaceToDepth ; Resize nearest et linéaire ; GridSample bilinéaire (align_corners, border et zeros) ; Pad ; ReduceMean et AveragePool. Tout opérateur non couvert par un noyau = refus nommé au build, jamais à l'exécution.
4. Familles de GPU : détection par identifiant constructeur et appareil (environ huit familles : NVIDIA Turing, Ampere, Ada, Blackwell ; AMD avant RDNA3, RDNA3, RDNA4 ; Intel Alchemist et Xe2+), liste d'autorisation des unités matricielles (VK_KHR_cooperative_matrix, VK_NV_cooperative_matrix2). La RTX de ce PC les a : écris et teste pour de vrai la variante matricielle de Conv (FP16, tuiles choisies d'après le profil mesuré de la RTX), derrière une interface de variante, et mesure-la contre la variante générique ; sur lavapipe (CI), seule sa non-sélection se teste.
5. Autotuner : chronomètre les variantes candidates par forme de couche (requêtes d'horodatage), garde la meilleure dans un cache par GPU, pilote, modèle et résolution ; tables pré-réglées possibles par famille (structure prête, tables vides).
Tests sur la RTX (et sur lavapipe en CI) : chaque opération contre l'interpréteur CPU de gpu_ir (tolérance FP32 1e-4 relatif ; FP16 plus large, justifiée), petits graphes façon SPAN (conv 3x3 + PReLU + pixel shuffle, x2) et façon RIFE (GridSample, Concat, Resize) de bout en bout, aller-retour de l'Artifact, command buffer rejoué, KernelOps contre la référence kernels/ref, chargement de la bibliothèque par dlopen avec la bonne et une mauvaise ABI, refus nommé d'un opérateur inconnu. Mesures (sous le verrou GPU) : SPAN AnimeJaNai réel (cherche les .onnx comme le chantier tensorrt : ~/.local/share/AnimeJaNai-HDR-linux…) à 1080p vers 4K et RIFE 4.26 à 720p, comparés aux chiffres TensorRT de ~/ajp-reports/tensorrt.md s'il existe ; le plan vise 1,0 à 1,4 fois le temps de TensorRT.`,
  },
  {
    key: 'gpu', branch: 'pc/gpu', base: 'w1/integration', wait_fondations: true,
    scope: 'src/gpu/ hors src/gpu/vulkan/ et src/gpu/ir/ (la bibliothèque ajp_gpu devient STATIC avec ses sources), nouveau backend de référence src/gpu/cpu/ (bibliothèque partagée ajp_backend_cpu, MODULE), et tests/unit/gpu/',
    brief: `Le module gpu : il exécute le Plan. Lis dans le plan : « Le cœur résout, le GPU exécute », la table des ordonnancements S0 à S3, « Calculer moins plutôt que plus vite », et la boîte de gpu.
1. Chargeur de backends : charge ajp_backend_<nom> (.so, .dll) depuis un dossier donné (celui des moteurs installés par l'utilisateur, passé par app), vérifie l'ABI, crée le backend avec BackendInit, le détruit par RAII ; liste les backends présents d'un dossier pour le panneau des capacités ; refus nommés (bibliothèque absente, symbole absent, ABI différente, appareil refusé).
2. Backend CPU de référence (ajp_backend_cpu) : Backend et KernelOps sur l'hôte, avec kernels/ref et l'interpréteur de gpu_ir. Il ne sera pas proposé à l'utilisateur ; il sert aux tests de bout en bout et au débogage. Documente-le comme tel.
3. Program (implémente ProgramBase) : construit depuis un Plan et des moteurs chargés ; décrit les étapes par ordonnancement : ingestion, réduction éventuelle, niveaux d'interpolation S1 avant l'agrandissement, passes d'agrandissement, niveaux S2 après, S3 (mouvement à basse résolution, warp et mélange à la résolution de sortie), écriture dans le ring. La sortie d'un niveau devient la source du suivant sous des SourceId dérivés.
4. Exécuteur : par image décodée, l'accumulateur de cadence de core donne les instants de sortie de chaque niveau ; la timeline des dessins de core décide REPEATED (Fast) et la réutilisation d'un doublon exact (Quality) ; la coupe de scène supprime l'interpolation de la paire (répétition de la source) ; Flush et époque vident sans attente hôte ; un Install arrive par Command et s'applique à la paire suivante sans Flush si ses moteurs sont prêts. Statistiques pour l'OSD : temps par paire, host_waits, images REPEATED, coupes.
5. Ring : états des slots (libre, en écriture, prêt, affiché), borné, rendu par SlotFree ; quand aucun slot n'est libre, le thread attend SlotFree dans sa boîte.
6. Thread gpu : boîte bornée (4 DecodedFrame, taille du ring pour SlotFree, Install, Command, Flush), Credit rendu au producteur, RingFrame poussé vers le thread main par une interface de puits, démarrage par l'utilitaire de core, arrêt par stop_token.
Tests de bout en bout avec le backend CPU : petites images synthétiques (64x36, YUV420P hôte), petits modèles ONNX construits dans le test (agrandissement x2 façon SPAN, interpolation au contrat d'entrée RIFE de kernels/spec), vérification des tailles, de la suite des pts pour x2, x3 et 5/2, de REPEATED sur dessins tenus, de la coupe de scène, du Flush, de la contre-pression du ring et du Credit, d'un Install en cours de flux, des SourceId dérivés ; chargeur testé avec une vraie bibliothèque construite et avec une bibliothèque absente ou d'une autre ABI. Tests du thread sous TSan (preset linux-prefix-tsan).`,
  },
  {
    key: 'render', branch: 'pc/render', base: 'w1/integration',
    scope: 'src/render/ et tests/unit/render/ (et un dossier de shaders texte si besoin, par ex. src/render/shaders/)',
    brief: `Le module render : composition libplacebo, grade, sous-titres, dessin de l'OSD ImGui ; hors écran en export. Lis dans le plan : « HDR : un grade sur deux routes, réglé sur l'écran », la table des paramètres du grade, « Export : le même Plan, sans horloge » et « Interface ».
1. RenderDevice : pl_log relié au canal de journal du module, pl_vk_inst avec les extensions d'instance fournies par l'appelant (celles de SDL pour une fenêtre ; aucune en mode sans écran), pl_vulkan_create avec les extensions et fonctions que demandent le décodage Vulkan de FFmpeg (lis hwcontext_vulkan.h et la façon dont mpv les passe : video/out/vulkan/ du fork) et notre moteur (FP16, stockage 16 bits, unités matricielles si présentes, mémoire et sémaphores externes pour l'interop CUDA). Remplit le GpuDevice du contrat (poignées, files, verrouillage des files par pl_vulkan lock_queue/unlock_queue). Mode sans écran pour l'export et les tests.
2. Slots du ring : alloue les slots (pl_tex RGBA16 stockables, échantillonnables, exportables par descripteur ou HANDLE) et leurs sémaphores timeline, remplit RingSlotDesc (pl_vulkan_unwrap), applique le protocole hold/release (pl_vulkan_release_ex, pl_vulkan_hold_ex) pour que gpu écrive et que render lise sans copie, et rend SlotFree avec la valeur de fin de lecture.
3. Composition : un RingFrame devient une pl_frame (représentation et espace selon le Plan et la source), pl_render_image vers une cible fournie par l'appelant (pl_frame : celle de la swapchain de present, ou une texture hors écran). render ne touche jamais à la swapchain ; present ne fait jamais pl_render_image : app reliera les deux. Mise à l'échelle vers la fenêtre, espace de sortie selon DisplayInfo (PQ BT.2020 si HDR actif, SDR sinon).
4. Grade : reprend anime-hdr-grade.hook du lecteur actuel (/home/claude/ta2b/mpv-animejanai-hdr/BuildMpvUpscale2xAnimeJaNai/mpv-upscale-2x_animejanai/portable_config/user-shaders/anime-hdr-grade.hook, et anime-hdr-calibration.hook), chargé par pl_mpv_user_shader_parse ; copie-le dans le dépôt avec son origine. Paramètres du grade depuis le profil et DisplayInfo (peak, abl, paper plafonné à 0,75 × peak ou 0,5 × peak avec TrueHDR et son émulation, sdr_white 203, shadows, highlights, vibrance, contraste, saturation, gris moyen, black_lo, black_hi) passés par les paramètres du hook ; dixième passe du vrai noir OLED décrite dans le plan (pied smoothstep C1, gain commun à R, G, B, max local 8x8, sautée si black_hi <= black_lo) si le hook ne l'a pas ; bascule A/B ; grade d'une source déjà HDR seulement si grade_native_hdr.
5. Sous-titres : le SubtitleFrame de subs (atlas R8, rectangles, couleurs) est téléversé dans une pl_tex R8 réutilisée (seulement s'il a changé), puis dessiné après le grade en pl_overlay monochrome coloré, au blanc subtitle_white (200 nits par défaut) ; l'atlas est rendu à subs après usage.
6. Dessin ImGui : une fonction qui dessine un ImDrawData par pl_dispatch_vertex (texture de police, rectangles de découpe), en dernier, par-dessus l'image ; le contexte ImGui et l'interface appartiennent à app.
7. Hors écran : cible pl_tex RGBA16 PQ BT.2020 à la taille de sortie, grade compris, sans OSD, lue de façon asynchrone (pl_buf) pour l'export.
Tests (sans écran, sur la RTX ; ils tourneront aussi sur lavapipe en CI) : appareil créé et GpuDevice rempli ; slot exporté puis réimporté ; un slot RGB connu rendu hors écran en SDR redonne ses pixels dans la tolérance ; une cible PQ code le blanc 203 nits à la valeur attendue ; les sous-titres apparaissent au bon endroit, de la bonne couleur, au bon niveau ; grade activé ou non change l'image et reste déterministe ; un ImDrawData simple dessine un rectangle au bon endroit ; le hook se charge. Mesure (sous le verrou GPU) le coût du grade et de la composition par image 4K sur la RTX (le plan : 1,4 ms sur D3D11, 2,0 ms de rendu par image 4K).`,
  },
  {
    key: 'present', branch: 'pc/present', base: 'w1/integration',
    scope: 'src/present/ et tests/unit/present/',
    brief: `Le module present : fenêtre, swapchain, cadence, détection de l'écran. Lis dans le plan : « Lecture : décodage matériel, horloge choisie par l'écran », la table des cadences par écran, la table de lecture de l'écran par système et l'ordre des indices OLED.
1. Fenêtre SDL3 (SDL_WINDOW_VULKAN, haute densité, plein écran), extensions d'instance requises (SDL_Vulkan_GetInstanceExtensions) remises à render, surface créée sur l'instance de render (SDL_Vulkan_CreateSurface), pl_swapchain par pl_vulkan_create_swapchain en FIFO, redimensionnement, indice d'espace couleur (pl_swapchain_colorspace_hint) pour HDR10 PQ avec métadonnées dont max_luma égal au pic du grade ; sans PQ, repli scRGB et grade coupé (expose l'information). present fournit à chaque image une cible pl_frame (pl_swapchain_start_frame) puis la soumet ; il ne fait jamais pl_render_image (render compose dans la cible, app relie les deux).
2. Détection de l'écran : DisplayInfo depuis SDL3 (mode courant et rafraîchissement exact en rationnel, HDR actif, blanc SDR, marge HDR), chaque champ forçable par le profil, avec sa provenance pour l'OSD ; le blanc de référence du système est proposé, jamais appliqué en silence. Windows (DXGI_OUTPUT_DESC1, DisplayConfigGetDeviceInfo GET_SDR_WHITE_LEVEL) derrière #ifdef _WIN32 ; Linux sans libdisplay-info (pas une dépendance) : note ce qui manque ; VRR inconnu par SDL : vient des réglages.
3. Horloge d'affichage : estimation robuste de la période de vsync depuis les instants de présentation en FIFO (filtre résistant aux images manquées), exposée comme horloge maître quand l'écran est fixe ; écart mesuré entre horloge audio et écran exposé pour le rééchantillonnage de l'audio (la borne de 0,125 % est appliquée par audio).
4. Ordonnanceur de présentation (logique pure, le cœur du module, très testé) : à partir des RingFrame prêts (pts, époque, drapeaux) et de l'horloge maître (MasterClock de core/clock.h : audio avec VRR, écran sinon), choisit l'image à montrer à chaque vsync (écran fixe, grille des vsyncs) ou à son horodatage (VRR), compte images sautées et répétées, gère pause, pas image, seek (époque), images en retard ; rend SlotFree avec la valeur du sémaphore de fin de lecture.
Tests : simulations de l'ordonnanceur (23,976 im/s vers 120 Hz avec x5 et 5/3, 144 Hz x3 à 2 vsyncs par image, 165 Hz 11011/3200, 60 Hz 5/2, VRR, dérive, retards, seek, pause, pas image) ; estimation de la période sur des instants synthétiques bruités ; DisplayInfo forcé et provenance ; test de fumée sur le vrai écran de ce PC (KDE Plasma sous Wayland, fenêtre petite et brève) : fenêtre, swapchain, quelques images effacées présentées, redimensionnement, période de vsync mesurée, et lecture de l'état HDR de l'écran ; le même test se SKIP proprement sans surface (CI). Note ce que l'écran réel annonce (rafraîchissement, HDR, VRR) dans ~/ajp-reports/display.md.`,
  },
  {
    key: 'export', branch: 'pc/export', base: 'w1/integration',
    scope: 'src/export/ et tests/unit/export/',
    brief: `Le module export : encodage, mux et métadonnées HDR10, sans horloge. Lis dans le plan « Export : le même Plan, sans horloge » et la table des encodeurs.
1. Encodeur (libavcodec) choisi par nom avec refus nommé s'il manque à la construction de FFmpeg : ffv1 (sans perte, tests et archivage), hevc_nvenc et av1_nvenc 10 bits (testés pour de vrai sur la RTX ; refusés proprement sans GPU NVIDIA), libsvtav1 si présent, encodeurs Vulkan si présents. Formats 10 bits (yuv420p10, p010) et RGB 16 bits, étiquettes de couleur (BT.2020, PQ), options de débit et qualité avec plages et refus.
2. Entrée : images RGBA16 PQ BT.2020 avec pts, lues depuis la sortie hors écran de render par une interface simple (ExportFrameSource) ; conversion hôte vers le format de l'encodeur par libswscale pour la voie logicielle (la référence kernels/ref ne part jamais dans un binaire livré) ; la voie GPU (noyau d'écriture P010 lu par l'encodeur matériel sans copie hôte) est décrite par l'interface et laissée au backend.
3. Mux (libavformat, MKV et MP4) : la vidéo encodée, plus la copie des pistes audio et sous-titres, des chapitres et des pièces jointes (polices) du fichier source, horodatages recalés ; sous-titres gravés seulement sur demande (render les grave alors).
4. HDR10 : métadonnées de mastering (pic = pic visé par le grade) et de niveau de lumière ; MaxCLL et MaxFALL mesurés pendant l'encodage et écrits au remux final (deuxième passe de mux si le conteneur l'exige).
5. Progression (images faites et totales, débit en im/s, temps restant), annulation, thread export (boîte bornée, Credit, démarrage par l'utilitaire de core) et une fonction run_export(ExportJob) utilisable sans fenêtre, que la ligne de commande ajp export appellera (vague 3).
6. Déterminisme : deux exports identiques donnent le même framemd5 (fonction de calcul exposée, utilisée par ajp bench --dump plus tard).
Tests : export ffv1 MKV d'images synthétiques relu et comparé sans perte ; métadonnées HDR10 relues (side data de codecpar) ; audio, sous-titres et chapitres d'une source synthétique créée dans le test recopiés ; MaxCLL et MaxFALL justes sur des images connues ; progression et temps restant ; annulation ; refus (encodeur inconnu, taille impaire en 4:2:0, option hors plage) ; framemd5 identique sur deux exports ; export NVENC HEVC 10 bits réel sur la RTX relu et vérifié (étiquettes, métadonnées HDR10), avec son débit en im/s.`,
  },
]

const RESULT = {
  type: 'object',
  properties: {
    branch: { type: 'string' },
    commit: { type: 'string' },
    summary: { type: 'string', description: 'ce qui a été implémenté, en français, 5 à 20 lignes' },
    tests: { type: 'string', description: 'commande et résultat des tests (nombre passés/échoués/sautés)' },
    checks: { type: 'string', description: 'clang-format, clang-tidy, tailles de fichiers, sanitizers : résultat' },
    shared_changes: { type: 'array', items: { type: 'string' }, description: 'fichiers modifiés hors du périmètre, et pourquoi' },
    open_questions: { type: 'array', items: { type: 'string' } },
    not_done: { type: 'array', items: { type: 'string' }, description: 'ce qui reste, et pourquoi' },
  },
  required: ['branch', 'commit', 'summary', 'tests', 'checks', 'shared_changes', 'open_questions', 'not_done'],
}

const REVIEW = {
  type: 'object',
  properties: {
    branch: { type: 'string' },
    commit: { type: 'string' },
    verdict: { type: 'string', enum: ['ok', 'fixed', 'needs_work'] },
    issues_found: { type: 'array', items: { type: 'string' } },
    fixes_applied: { type: 'array', items: { type: 'string' } },
    tests: { type: 'string' },
    remaining: { type: 'array', items: { type: 'string' } },
  },
  required: ['branch', 'commit', 'verdict', 'issues_found', 'fixes_applied', 'tests', 'remaining'],
}

const INTEGRATION = {
  type: 'object',
  properties: {
    branch: { type: 'string' },
    commit: { type: 'string' },
    merged: { type: 'array', items: { type: 'string' } },
    conflicts: { type: 'array', items: { type: 'string' }, description: 'conflits rencontrés et leur résolution' },
    contract_mismatches: { type: 'array', items: { type: 'string' }, description: 'désaccords entre modules et correction' },
    e2e: { type: 'string', description: 'tests de bout en bout ajoutés et résultat' },
    tests: { type: 'string' },
    remaining: { type: 'array', items: { type: 'string' } },
  },
  required: ['branch', 'commit', 'merged', 'conflicts', 'contract_mismatches', 'e2e', 'tests', 'remaining'],
}

phase('Implémenter')
const results = await pipeline(
  MODULES,
  m => agent(
    `${COMMON}\n\nTon module : ${m.key}\nBase : origin/${m.base} (git fetch origin ; git checkout -b ${m.branch} origin/${m.base})\nBranche : ${m.branch}\nPérimètre : ${m.scope}\n${m.wait_fondations ? 'Ce module dépend entièrement des contrats : tant que origin/w2/fondations n\'existe pas, lis le plan, le code de core et de gpu_ir, prépare ta conception, et revérifie toutes les dix minutes ; fusionne-la dès qu\'elle paraît.\n' : ''}\n${m.brief}\n\nTravaille jusqu'à ce que tout se construise sans avertissement et que tous les tests passent, puis pousse ta branche (git push -u origin ${m.branch}) et exécute git checkout --detach. Rends le résultat demandé.`,
    { label: `impl:${m.key}`, phase: 'Implémenter', isolation: 'worktree', schema: RESULT }),
  (impl, m) => impl ? agent(
    `${COMMON}\n\nTu es relecteur adverse du module « ${m.key} », écrit par un autre agent sur la branche locale ${impl.branch} (commit ${impl.commit}). Exécute « git checkout ${impl.branch} » dans ton worktree (au lieu de créer une branche), puis continue sur cette branche.\n\nCe qu'on attendait :\n${m.brief}\n\nPérimètre : ${m.scope}\n\nRésumé de l'auteur : ${impl.summary}\nChangements hors périmètre : ${JSON.stringify(impl.shared_changes)}\nCe qu'il dit ne pas avoir fait : ${JSON.stringify(impl.not_done)}\nSes questions : ${JSON.stringify(impl.open_questions)}\n\nTon angle : la justesse. Cherche ce qui est faux ou manquant, pas confirmer. Construis et lance les tests toi-même, y compris sous ASan/UBSan (cmake --preset linux-prefix-asan, build, ctest ; le pilote NVIDIA signale des fuites internes à libcuda et libnvidia-* : distingue-les des nôtres, voir le fichier de suppressions de pc/media-gpu s'il est poussé, et documente toute suppression) et sous TSan pour les threads. Relis chaque fichier contre la demande et les règles du dépôt : refus nommés, aucune exception qui sort, aucun getenv, synchronisation GPU correcte (barrières, sémaphores, durées de vie, ordre de destruction), bornes, aucune copie hôte par image sur le chemin de lecture. Vérifie que les tests testent vraiment le comportement (pas des tautologies) et que le code repris l'a été fidèlement. Corrige, complète ce qui manque et qui est faisable ici, ajoute les tests qui manquent, puis commite sur la même branche (règles Git de regles.md), pousse-la et exécute « git checkout --detach ». Rends le verdict.`,
    { label: `relire:${m.key}`, phase: 'Relire', isolation: 'worktree', schema: REVIEW }).then(r => ({ module: m, impl, review: r })) : null,
  prev => prev && prev.review ? agent(
    `${COMMON}\n\nTu es le second relecteur adverse du module « ${prev.module.key} » (branche locale ${prev.review.branch}, commit ${prev.review.commit}). Exécute « git checkout ${prev.review.branch} » dans ton worktree, puis continue sur cette branche.\n\nCe qu'on attendait :\n${prev.module.brief}\n\nLe premier relecteur (angle justesse) a trouvé : ${JSON.stringify(prev.review.issues_found)}\nIl laisse : ${JSON.stringify(prev.review.remaining)}\n\nTon angle est différent : la conformité au plan et la clarté. Relis la demande point par point et le plan (${B}/plan.txt) : chaque point est-il réellement fait, ou seulement esquissé ? Le code est-il limpide pour un lecteur qui découvre le module (noms, découpage, contrats écrits en tête des en-têtes, pas de code mort, pas de duplication avec un autre module, pas de bascule cachée) ? Les performances voulues par le plan sont-elles possibles avec cette conception (command buffers rejoués, pas d'allocation ni de copie hôte par image, pas d'attente hôte cachée) ? Les contrats partagés (${CONTRACTS}) sont-ils respectés à la lettre, de sorte que les quatre autres modules, écrits sans se parler, s'y brancheront ? Corrige et complète, refusionne origin/w2/fondations si elle a changé, puis commite sur la même branche (règles Git de regles.md), pousse-la et exécute « git checkout --detach ». Rends le verdict.`,
    { label: `relire-plan:${prev.module.key}`, phase: 'Relire', isolation: 'worktree', schema: REVIEW }).then(r => ({ key: prev.module.key, impl: prev.impl, review: prev.review, review2: r })) : prev,
)

phase('Intégrer')
const done = results.filter(Boolean).map(r => ({ key: r.key ?? r.module?.key, branch: (r.review2 ?? r.review ?? r.impl)?.branch }))
const integration = await agent(
  `${COMMON}\n\nTu es l'intégrateur de la vague 2 sur ce PC. Attends que origin/w2/fondations existe (git ls-remote ; revérifie toutes les dix minutes). Crée la branche pc/w2-integration depuis origin/w2/fondations, puis fusionne une à une (git merge --no-ff) : origin/w2/vk-compute, origin/pc/vk-compute-nvidia si elle existe, puis ces branches : ${JSON.stringify(done)}. Les conflits additifs (enum RefusalCode de core/types.h, listes SOURCES des CMakeLists.txt, add_subdirectory) se résolvent en gardant les deux côtés : ${B}/union_merge.py <fichier> le fait pour ces cas (lis-le d'abord) ; les autres conflits se résolvent à la main, en comprenant les deux côtés. Après chaque fusion : construction et tests complets.\n\nEnsuite : les cinq modules ont été écrits en parallèle sur les mêmes contrats. Vérifie qu'ils s'emboîtent vraiment : un test d'intégration de bout en bout sans écran (tests/integration/ ou tests/unit/e2e/) qui enchaîne media (un petit clip encodé dans le test) → gpu (backend Vulkan sur la RTX, et backend CPU) avec un Plan réel résolu par core (agrandissement x2 par un petit modèle ONNX construit dans le test, puis interpolation x2 Fast) → slots du ring alloués par render → composition hors écran par render (grade compris) → export ffv1, puis relit le fichier exporté et vérifie le nombre d'images, les pts et des pixels plausibles ; un second test avec present sur le vrai écran qui affiche quelques images du ring dans une vraie swapchain (SKIP sans surface) ; et une démonstration manuelle hors tests : un vrai épisode d'anime trouvé sur le PC, agrandi par le vrai SPAN sur notre backend Vulkan, affiché quelques secondes dans une fenêtre, avec les temps par image dans ~/ajp-reports/w2-demo.md. Tout désaccord entre modules (types, sémaphores, formats, propriété des objets, ordre de destruction) se corrige au bon endroit, avec un test. Lance enfin toute la suite sous ASan/UBSan et TSan, clang-tidy sur src/, les vérifications de taille et de format. Commite (règles Git de regles.md), pousse pc/w2-integration, puis git checkout --detach. Rends le résultat.`,
  { label: 'integrer:vague-2', phase: 'Intégrer', isolation: 'worktree', schema: INTEGRATION })

return { modules: results, integration }
