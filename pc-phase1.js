export const meta = {
  name: 'pc-phase-1',
  description: 'PC : media sur GPU, socle CUDA, TensorRT, TrueHDR réel, coquille de app',
  whenToUse: 'Sur le PC de Tatoub, en parallèle de la vague 2 du cloud',
  phases: [
    { title: 'Implémenter', detail: 'un agent par chantier, dans son worktree' },
    { title: 'Relire', detail: 'relecture adverse, correction, push' },
  ],
}

// args.briefs : chemin du checkout de la branche agents/briefs sur le PC (par ex. ~/ajp-briefs)
const B = args.briefs
const COMMON = `
Lis d'abord ${B}/regles.md en entier : il fixe la machine, les dépôts de référence, les règles du code et Git. Puis lis dans le dépôt docs/ARCHITECTURE.md, CMakeLists.txt, cmake/*.cmake, src/core/*.h, src/gpu/backend.h et le CMakeLists.txt des modules que tu touches. Le plan du projet est ${B}/plan.txt ; les comptes rendus de la vague 1 sont ${B}/w1-results.txt.
Les rapports déjà faits sur ce PC (inventaire, deux correctifs de tests media, questions de conception) sont dans ~/ajp-reports/2026-10-19-w1-integration/.
Pendant que tu travailles, d'autres agents travaillent sur ce PC et dans le cloud : le cloud écrit en ce moment les fondations de la vague 2 (branche w2/fondations : contrats GpuDevice, backend ABI 2 avec KernelOps, ring, démarrage des threads, journal) et le socle Vulkan (w2/vk-compute). Reste dans ton périmètre ; si tu dois toucher un fichier partagé, fais le plus petit changement additif et signale-le dans shared_changes.
`

const TASKS = [
  {
    key: 'media-gpu', branch: 'pc/media-gpu', base: 'w1/integration',
    scope: 'src/media/ et tests/unit/media/ (plus tests/sanitizers/ pour un fichier de suppressions LSan et son branchement dans CMakePresets.json)',
    brief: `Module media sur le vrai GPU, et les défauts trouvés sur ce PC.
1. Applique les deux correctifs de tests du rapport (decoder_test.cpp, seek_test.cpp), relus.
2. Défaut de conception A du rapport : dans SourceLoop, une file audio ou sous-titres pleine arrête la vidéo. Corrige-le : jamais de blocage sur une file annexe tant que la première image de l'époque n'est pas livrée (réserve FIFO plafonnée en octets, vidée au Flush et au Seek, refus nommé au dépassement) ; files annexes bornées en durée et en octets plutôt qu'en nombre de paquets si c'est plus juste ; contrat écrit (les consommateurs ne conditionnent jamais la lecture de leur file à la vidéo) ; threads de décodage logiciel plafonnés ou décodage synchrone si c'est plus sûr. Test de régression avec threads = 16 explicite, qui échouerait sans le correctif même sur 4 vCPU.
3. Défaut B : par défaut, Vulkan ne décode que les codecs Vulkan Video (H.264, HEVC, AV1, VP9) ; les décodeurs par compute shaders (ffv1_vulkan, prores_vulkan…) restent possibles par une option explicite. Tests.
4. Défaut C : libcuda reste chargée après cuInit (pas de dlclose) ; documente pourquoi.
5. NVDEC réel sur la RTX : ouverture CUDA avec le contexte primaire de l'appareil choisi par UUID (AV_CUDA_USE_PRIMARY_CONTEXT), copie de sécurité, extra_hw_frames dans get_format, comportement du pool (images tenues, libération), pas de ligne et native_plane cohérents ; décodage Vulkan Video H.264/HEVC/AV1 sur la RTX. Les clips de test se créent dans le test avec les encodeurs GPU disponibles (h264_nvenc, hevc_nvenc, av1_nvenc, ou h264_vulkan…) ; sans GPU NVIDIA, ces tests se SKIP avec une raison.
6. Suppressions LSan pour les fuites internes au pilote NVIDIA (libcuda, libnvidia-*, libdbus) dans un fichier versionné, branché sur le preset ASan ; les tests [vulkan] sur NVIDIA documentent le cas de l'ICD déchargé.
Lance toute la suite media sous ASan/UBSan et TSan, et répétée (--repeat until-fail:10) en -j16.`,
  },
  {
    key: 'cuda-compute', branch: 'pc/cuda-compute', base: 'w1/integration',
    scope: 'nouveau dossier src/gpu/cuda/ (src/gpu/cuda/CMakeLists.txt et src/gpu/cuda/compute/ : bibliothèque STATIC ajp_cuda_compute), add_subdirectory(cuda) dans src/gpu/CMakeLists.txt sous if(AJP_WITH_CUDA), et tests/unit/cuda_compute/',
    brief: `Socle de calcul CUDA : le port CUDA des noyaux de kernels/spec.h, base du futur backend TensorRT. Reprends les noyaux prouvés de animejanai-inference-rtx@118c272 src/kernels.cu (et src/resample.h, src/aji_trt.cpp pour l'usage), réécrits proprement sur la spec.
1. CMake : AJP_WITH_CUDA=ON active le langage CUDA (CUDA 13.4 dans /opt/cuda), la liste d'architectures du plan pour une livraison (75 à 120) et 89 en développement local (lis la note sur CMAKE_CUDA_ARCHITECTURES dans le CMakeLists.txt racine : une valeur explicite est obligatoire).
2. Contexte : contexte primaire de l'appareil choisi par UUID, flux non bloquants, événements, mémoire de l'appareil bornée (pool simple, refus nommé si elle manque).
3. Noyaux : ingestion depuis la mémoire linéaire à pas de ligne de NVDEC (NV12, P010, P016, et les formats planaires) vers RGB planaire FP16 et FP32, avec matrices, plages et siting ; rééchantillonnage Spline36 séparable, Area, bilinéaire ; padding mod-64 et recadrage RIFE ; warp arrière et mélange à l'instant t ; écriture vers RGBA16 (surface ou tableau CUDA, pour l'interop avec une image Vulkan importée) et vers P010 avec l'arrondi exact de la référence ; fonctions de transfert et primaires ; coupe de scène et différence par bloc 32x32 avec lecture asynchrone de 4 octets servie à la paire suivante (comme k_flow_max).
4. Interop : import d'une mémoire et d'un sémaphore Vulkan exportés (cuImportExternalMemory, cuImportExternalSemaphore, descripteurs opaques), tel que render exportera les slots du ring ; test avec une image Vulkan créée et exportée par le test sur la RTX.
Tests sur la RTX : chaque noyau contre la référence CPU de kernels/ref (tolérances de spec.h ; exact là où la référence l'est), bords et tailles refusées, lecture asynchrone, interop Vulkan vers CUDA ; tous [cuda], SKIP sans GPU. Mesure (sous le verrou GPU) le temps de chaque noyau à 1080p et 4K et compare à ce que le plan suppose.`,
  },
  {
    key: 'tensorrt', branch: 'pc/tensorrt', base: 'w1/integration',
    scope: 'src/gpu/cuda/trt/ (bibliothèque STATIC ajp_cuda_trt ; n\'écris pas src/gpu/cuda/CMakeLists.txt, qui appartient au chantier cuda-compute : crée-le seulement s\'il n\'existe pas dans ta base, avec juste add_subdirectory(trt), le conflit se résoudra à l\'intégration), tests/unit/cuda_trt/, et une installation du SDK hors dépôt dans ~/ajp-sdk/',
    brief: `TensorRT sur la RTX : le plan vise TensorRT 11.3 (version du fork), CUDA 13.4.
1. Installe TensorRT 11.x pour Linux x86_64 et CUDA 13 dans ~/ajp-sdk/tensorrt, sans sudo : bibliothèques depuis PyPI (paquets tensorrt-cu13 / tensorrt-cu13-libs, dans un venv, puis copie des .so) ou depuis les publications GitHub de NVIDIA ; en-têtes publics depuis le dépôt NVIDIA/TensorRT (étiquette de la même version) ; trtexec construit depuis samples/trtexec de ce dépôt s'il n'est pas livré. Note précisément versions et provenances. Si TensorRT 11 est introuvable sans compte NVIDIA, prends la 10.x la plus récente disponible et dis-le.
2. Enveloppe d'exécution (src/gpu/cuda/trt) : chargement d'un moteur sérialisé, IExecutionContext, profils de formes min/opt/max, entrées et sorties nommées, enqueueV3 sur un flux non bloquant, événements, journal TensorRT relié à spdlog, refus nommés. Le contrat Backend du cloud change en ce moment (ABI 2) : garde une interface interne propre, l'adaptation au contrat se fera après.
3. Construction réelle par le constructeur trtexec du module prepare (src/prepare) sur les vrais modèles : SPAN AnimeJaNai (cherche les .onnx dans ~/.local/share/AnimeJaNai-HDR-linux, ~/.local/share/AnimeJaNai, ~/Documents/mpv-AnimeJaNai-HDR ; ne commite aucun modèle) et RIFE 4.26 (même recherche ; sinon la source que le lecteur actuel utilise pour le télécharger), FP16, formes, îlots FP32 des GridSample de RIFE comme aji_trt.cpp, cache de timing. Corrige le constructeur de prepare s'il se trompe face au vrai trtexec (lignes de commande, analyse de la progression), avec tests.
4. Mesures (sous le verrou GPU, flock ~/ajp-reports/gpu-bench.lock) : temps par image de SPAN (1080p vers 4K et 720p vers 1440p) et de RIFE 4.26 à 720p, comparés aux chiffres du plan (« Calculer moins plutôt que plus vite », table de la 4070 Ti) ; temps de construction des moteurs. Écris le rapport dans ~/ajp-reports/tensorrt.md.
Tests [cuda] : chargement d'un moteur construit dans le test depuis un petit ONNX généré dans le test, exécution et comparaison à l'interpréteur CPU de gpu_ir ; refus nommés (moteur corrompu, forme hors profil).`,
  },
  {
    key: 'vendor-hdr-real', branch: 'pc/vendor-hdr-real', base: 'w1/integration',
    scope: 'src/vendor_hdr/ et tests/unit/vendor_hdr/',
    brief: `TrueHDR réel : le SDK Maxine VFX est installé sur ce PC (/usr/local/VideoFX). La vague 1 a écrit les chargeurs sans le SDK ; la relecture a laissé des points non vérifiés (lis la fin du compte rendu vendor-hdr dans ${B}/w1-results.txt) : noms des bibliothèques, valeurs des énumérations NVCV (NVCV_RGB10A2, NVCV_P32…), chaînes de sélection des effets, marqueur de version, signature des fonctions.
1. Vérifie chaque prototype, constante et nom contre les vrais en-têtes et bibliothèques du SDK (et contre le chemin Linux du lecteur actuel : ~/Documents/animejanai-inference-rtx, aji-rtx-linux/, en lecture seule) ; corrige ce qui est faux et cite la source.
2. Fais tourner TrueHDR pour de vrai sur la RTX : création pour une taille et un format, paramètres contraste, saturation, gris moyen, pic (plages du plan), process sur des tampons CUDA, contrat de flux (flux 0 hérité, encadré d'événements), entrée 8 bits tramée avec la vraie matrice.
3. Mesures et questions du plan : temps par image à 1080p et 4K (sous le verrou GPU) ; déterminisme de TrueHDR (deux passages identiques donnent-ils les mêmes octets ?) ; faisabilité de la rampe de calibration de 32 marches quasi noires avec une seule instance ; comportement si on recrée l'instance (règle d'instance unique).
4. Magasin des SDK : import réel depuis /usr/local/VideoFX vers un dossier de données de test, vérification de version.
Tests [nvidia] qui SKIP sans SDK ni GPU ; rapport dans ~/ajp-reports/truehdr.md.`,
  },
  {
    key: 'app-shell', branch: 'pc/app-shell', base: 'w1/integration',
    scope: 'src/app/ (le module devient une bibliothèque avec ses sources, plus un exécutable ajp) et tests/unit/app/',
    brief: `Les parties de app qui ne dépendent pas encore de render, present ni gpu. Lis dans le plan « Interface : une fenêtre, un OSD qui dit la vérité », la section « Premier lancement : choisir ses moteurs » et les réglages.
1. Ligne de commande : ajp play <fichier>, ajp export <fichier> …, ajp bench, ajp import-sdk <dossier>, ajp engines (liste et état), options de DebugFlags (core/debug_flags) ; aide claire en français ; refus nommés.
2. Détection du système pour core::recommend_engines (SystemInfo de core/engines.h) : GPU par énumération Vulkan (loader chargé par dlopen, identifiants PCI constructeur et appareil, nom, pilote, UUID), présence des runtimes par sondage de bibliothèques (libcuda et version du pilote, ROCm/MIGraphX, OpenVINO), OS ; testée sur cette RTX et avec des sondes factices.
3. Téléchargement des paquets de moteurs et des modèles (libcurl, ajp::dep::curl) : reprise, progression, annulation, délais, vérification SHA-256 et taille par core/model_index, installation atomique dans le dossier de données, refus nommés ; testé contre un petit serveur HTTP local lancé par le test (pas d'accès réseau dans les tests).
4. Modèle de l'écran du premier lancement, indépendant de l'interface : liste des moteurs de recommend_engines avec badges (Recommandé, Seulement AMD, Seulement Intel, Compatibilité…), présélection, taille à télécharger et place libre, validation en un clic, moteurs grisés installables en mode avancé, réouverture depuis les réglages ; écrit dans settings (core/settings) ; la facilité de prise en main est l'exigence du propriétaire. Le dessin ImGui viendra quand render aura son dessin ImGui : sépare l'état et l'action du dessin.
5. Chemins des données utilisateur par OS (passés en paramètre partout, déterminés ici une seule fois sans getenv dans src/ : utilise les fonctions de SDL3, SDL_GetPrefPath ou équivalent).
Tests Catch2 pour chaque partie.`,
  },
]

const RESULT = {
  type: 'object',
  properties: {
    branch: { type: 'string' },
    commit: { type: 'string' },
    pushed: { type: 'boolean' },
    summary: { type: 'string', description: 'ce qui a été fait, en français, 5 à 20 lignes' },
    measurements: { type: 'string', description: 'chiffres mesurés sur la RTX, ou vide' },
    tests: { type: 'string' },
    checks: { type: 'string' },
    shared_changes: { type: 'array', items: { type: 'string' } },
    open_questions: { type: 'array', items: { type: 'string' } },
    not_done: { type: 'array', items: { type: 'string' } },
  },
  required: ['branch', 'commit', 'pushed', 'summary', 'measurements', 'tests', 'checks', 'shared_changes', 'open_questions', 'not_done'],
}

const REVIEW = {
  type: 'object',
  properties: {
    branch: { type: 'string' },
    commit: { type: 'string' },
    pushed: { type: 'boolean' },
    verdict: { type: 'string', enum: ['ok', 'fixed', 'needs_work'] },
    issues_found: { type: 'array', items: { type: 'string' } },
    fixes_applied: { type: 'array', items: { type: 'string' } },
    tests: { type: 'string' },
    remaining: { type: 'array', items: { type: 'string' } },
  },
  required: ['branch', 'commit', 'pushed', 'verdict', 'issues_found', 'fixes_applied', 'tests', 'remaining'],
}

const results = await pipeline(
  TASKS,
  t => agent(
    `${COMMON}\n\nTon chantier : ${t.key}\nBase : origin/${t.base}\nBranche : ${t.branch}\nPérimètre : ${t.scope}\n\n${t.brief}\n\nTravaille jusqu'à ce que tout se construise sans avertissement et que tous les tests passent, puis pousse ta branche. Rends le résultat demandé.`,
    { label: `impl:${t.key}`, phase: 'Implémenter', isolation: 'worktree', schema: RESULT }),
  (impl, t) => impl ? agent(
    `${COMMON}\n\nTu es relecteur adverse du chantier « ${t.key} », fait par un autre agent sur la branche ${impl.branch} (commit ${impl.commit}). git fetch origin, puis git checkout ${impl.branch} (la branche locale existe si le worktree la partage ; sinon git checkout -b ${impl.branch} origin/${impl.branch}).\n\nCe qu'on attendait :\n${t.brief}\nPérimètre : ${t.scope}\n\nRésumé de l'auteur : ${impl.summary}\nMesures : ${impl.measurements}\nChangements hors périmètre : ${JSON.stringify(impl.shared_changes)}\nCe qu'il dit ne pas avoir fait : ${JSON.stringify(impl.not_done)}\nSes questions : ${JSON.stringify(impl.open_questions)}\n\nCherche ce qui est faux ou manquant, pas confirmer. Construis et lance les tests toi-même, sous ASan/UBSan et TSan quand il y a des threads, sur la RTX. Relis chaque fichier contre la demande et les règles (refus nommés, aucune exception qui sort, aucun getenv, synchronisation GPU, bornes, aucune copie hôte par image sur le chemin de lecture). Vérifie que les mesures sont honnêtes (verrou GPU, échauffement, plusieurs passes, écart type) et que les tests testent vraiment le comportement. Corrige, complète, commite sur la même branche, pousse-la, puis git checkout --detach. Rends le verdict.`,
    { label: `relire:${t.key}`, phase: 'Relire', isolation: 'worktree', schema: REVIEW }).then(r => ({ task: t.key, impl, review: r })) : { task: t.key, impl: null, review: null },
)
return results
