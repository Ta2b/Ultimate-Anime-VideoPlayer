# Plan de la v0.6

Les retours de Tatoub du 28 octobre (après la v0.5.0, publiée en pré-version) forment la **v0.6**, pas une v0.5.1. La version CMake passe à 0.6.0 **à la
préparation de la release v0.6**, pas avant. Même règle qu'en v0.5 : une PR par étape, base main, clang-format avant chaque PR ; l'étiquette n'est posée
que sur un accord clair de Tatoub dans le fil. Détail des décisions et journal : ce fichier ; l'historique de la 0.5 : `plan-v0.5.md`.

## Étapes, dans l'ordre (état au 28/10, 15:20)
- [x] Catalogue RIFE réduit à 4.25 et 4.22 lite (#48) ; menu SPAN réduit au catalogue AnimeJaNai HD V3.1 (#55) : plus de profil « legacy » dans les listes
- [x] Fenêtre fantôme à la fermeture (Fedora, Wayland, GNOME) : correctif défendable par la lecture du code (#49) ; **à confirmer sur le Fedora GNOME**
- [x] Interpolation Auto par défaut pour tous les profils (#53)
- [ ] « Garder les poses DRBA » par défaut : **performance mesurée** (poses actives : tient, retard 0,0 ms, composition 1,2 ms ; inactives : ne tient pas, 15,1 ms, 6,0 ms ; 4070 Ti, Re:Zero 18, ×3) ; qualité : comparaison d'images en cours, puis décision  — **MAJ 15:15 : fait par défaut (#84)** ; qualité : captures équivalentes (aucun fantôme), pas de mesure objective
- [x] HDR : écran SDR sans conversion HDR (options grisées, note), mode SDR gradé pour imiter le rendu HDR, RTX HDR non installé : repli sur l'adaptatif (#68, #69) ; **à confirmer chez Tatoub**
- [x] Interface : style Showtime (commandes rondes au centre, titre en gras), fenêtre « lecteur par défaut » au démarrage, réglages avancés dans les onglets existants sans catégorie Avancé, Statistiques et Réinitialiser dans Général (#71, #72) ; reste le menu : molette, barre de défilement, libellés, disposition horizontale  — **MAJ 15:15 : molette sur l'arbre et 5 catégories (#87), barre de défilement fine (#89)** ; restent libellés français, disposition horizontale
- [ ] Benchmark autonome et silencieux : séquence de test écrite par le lecteur, fenêtre minuscule, étapes annoncées (#75) ; **reste** l'exploration des moteurs, multiplicateurs et résolutions, Auto en dernier  — **MAJ 15:15 : #75 fusionnée ; l'exploration reste à faire**
- [x] Moteurs : espace disque vérifié avant l'installation, dossier d'installation au choix (#54) ; installation TensorRT corrigée : architecture du GPU, progression unique, un seul essai (#60) ; paquet TensorRT découplé des étiquettes (#67)
- [x] Audio : profils de virtualisation (#64), distance, pièce, égaliseur, nuit, profil par appareil (#80), **à écouter** ; [ ] convolution HRIR à la HeSuVi (#82, en revue)  — **MAJ 15:15 : #82 fusionnée** (non écoutée)
- [x] Empaquetage : AppImage (#74, essai local #77), installeur .exe Inno Setup + zip portable (#74, **à valider par la prochaine release Windows**), .dmg (#79)
- [x] Police : repli CJK vérifié (#59), chargeur FreeType pour le CFF2 de Fedora (#70) ; **à confirmer chez Tatoub**
- [x] CI allégée (#73) ; dépôt des releases séparé **gardé** (décision de Tatoub, 28/10)
- [ ] Précision et accélération GPU (section « Précision et accélération GPU » plus bas) : crans Performance FP8 / Équilibre FP16 / Qualité FP32 (#88 en revue), tuiles FP8 Vulkan (#90), import Q/DQ FP8 dans gpu_ir (#91), coopmat RDNA 4 (#86) ; mesures du 28/10 faites (FP8 plus lent sur SPAN, −6 % sur RIFE, mixte qualité 53,7 dB), plugin SPAN fusionné, SPAN mixte a/b, ModelOpt, CUTLASS, GMFSS/DistilDRBA FP8 et NHWC en cours de mesure ; pas de QAT (décision)
- [ ] Release v0.6 : version 0.6.0, notes `tools/release/notes/v0.6.0.md` (brouillon #81), liste des essais Windows/macOS ; étiquette seulement sur le message direct de Tatoub

## Journal
- **2026-10-29 (soir)** : #118 à #131 et #134 fusionnées ; #133 (benchmark simple, SPAN FP8 automatique) ouverte ; v0.6.0 Linux publiée, échecs macOS et Windows corrigés ; plus de nouvelle fonctionnalité jusqu'à la 1.0 (audits en cours).
- **2026-10-29** : #104 à #115 fusionnées ou prêtes (voir la liste vivante, points 19 à 24). Paquet TensorRT Linux construit (trt-7b7b65cbe8d6, à refaire depuis la main finale). Restent avant l'étiquette : menus (captures), vérification du pop-up, pas d'étiquette sans « go » du coordinateur.
- **2026-10-28** : v0.6 ouverte. Mesures du contrôleur Auto sur la tête 00f00ee : RIFE ×4, DistilDRBA ×5, GMFSS ×2 tenus en fin de run (`docs/reports/mesures-auto-4070ti-2026-10-28.md`).
- [x] Paquet TensorRT découplé des étiquettes : empreinte de ses entrées (`trt-fingerprint.sh`), release fixe `trt-<empreinte>` (`trt-package.sh`), Release Linux la lit sans relance ; plus de reconstruction ni de seconde passe par release (demande de Tatoub, 0.5.1)

## Liste vivante du détail demandé par Tatoub (28/10) : état

1. Audio : FAIT (#64, #80) : profils, distance, pièce, égaliseur, nuit, profil par appareil ; à écouter.
2. Réglages avancés : FAIT (#61, #72) : dans les onglets existants, OLED automatique, Réinitialiser et Statistiques hors des avancés.
3. Pop-up « lecteur par défaut » et réglage discret : FAIT (#71) ; composition finale et soins OLED hors de Couleurs : FAIT (#72, dans Agrandissement et HDR avec les avancés).
4. HDR : FAIT (#68, #69) ; à confirmer chez Tatoub.
5. Benchmark autonome et silencieux : EN PARTIE (#75) ; reste l'exploration des moteurs, multiplicateurs et résolutions.  **MAJ 29/10 : exploration (moteurs, multiplicateurs, résolutions) toujours À FAIRE.**  **MAJ 29/10 (soir) : exploration FAITE (#119) ; « Évaluer ma machine » refait en un bouton « Optimiser pour ma machine » (PR #133 ouverte) : résultat enfin relu (bench.toml écrit au mauvais endroit), délais adaptés à la construction des moteurs, échecs expliqués.**
6. Poses DRBA par défaut : EN COURS : performance mesurée (favorable), qualité à comparer.  **MAJ 15:15 : FAIT (#84).**
7. Design Showtime FAIT (#71) ; AppImage FAIT (#74, #77) ; installeur .exe + zip portable FAIT (#74), à valider par la release Windows ; .dmg FAIT (#79).
8. Fedora GNOME : fenêtre fantôme (#49) et japonais (#59, #70) : À CONFIRMER chez Tatoub.
9. Convolution HRIR à la HeSuVi : EN REVUE (#82) ; préréglage « Voix d'anime » FAIT (#80).  **MAJ 15:15 : #82 fusionnée.**
10. Auto en VRR, demi-paliers ×0,5 : À FAIRE (trancher, mesurer, répondre).  **MAJ 29/10 : FAIT (#113)** : demi-paliers ×1,5, ×2,5… sur un écran VRR armé, paliers entiers sinon ; tests chiffrés ; mesure en lecture réelle à refaire machine libre.
11. Menu : molette, barre de défilement, moins de catégories, disposition horizontale, libellés français : À FAIRE (pastille « Le plus lent » FAITE).  **MAJ 15:15 : molette (#87), 5 catégories (#87), barre fine (#89) faites ; restent disposition horizontale et libellés.**  **MAJ 29/10 : arbre à gauche, retrait des pages, catégorie repliable (#109) ; en cours (branche menus-libelles) : rangée des moteurs de mouvement, note VRR courte, nom du GPU sur l’accueil, captures 1920×1080.**  **MAJ 29/10 (soir) : arbre redessiné, barre de lecture d'origine rétablie, fenêtre « lecteur par défaut » centrée (#118) ; crans de précision : seuls les crans utilisables apparaissent (#129) ; audit esthétique complet en cours (branche audit-ui).**
12. Arbre : EN PARTIE (#72) ; à revoir : place de Langues, Export, Moteurs, Couleurs sans réglages sans rapport.
13. « Ouvrir avec » enregistré au premier lancement : FAIT (assistant).
14. Config auto (tranche du GPU et résolution de l'écran, réduire puis agrandir, SPAN Sharp1 Performance, modèle par tranche) : FAIT (#53, tests `[auto_config]`).
15. Assistant : choix guidé du moteur : À FAIRE.  **MAJ 29/10 : vérifié dans first_run.cpp : FAIT, les huit écrans sautables (moteur recommandé par GPU, RTX HDR pas à pas, « Relancer l’assistant ») couvrent la spécification (ux-parametres.md 4.8).**
16. Poses DRBA : contradiction du tableau de décisions (« désactivées ») : à corriger après la décision (point 6).
17. .dmg FAIT (#79) ; archive TensorRT Windows manquante : À FAIRE (même schéma que Linux, #67) ; build macOS corrigé (#66).  **MAJ 29/10 : archive TensorRT Windows toujours manquante (limite connue : demande une machine Windows, le SDK TensorRT-RTX et attach-nvidia-windows.ps1) ; l’archive Linux est faite : release fixe trt-7b7b65cbe8d6, à refaire depuis la main finale (ABI 15).**  **MAJ 29/10 (soir) : v0.6.0 Linux publiée ; corrections macOS (#127, #128), Windows (#128, #131, #134) ; paquet TensorRT Linux publié pour l'empreinte courante (trt-caf273dd3732) ; la release v0.6.0 est à relancer pour l'inclure.**
18. Police japonaise en 0.5.1 : CORRIGÉE (#70), à confirmer chez Tatoub.

## Précision et accélération GPU : graphes spécialisés par cran (demande de Tatoub, 28/10)

**But.** Profiter au maximum des accélérations spécialisées des cartes récentes NVIDIA (Ada, Blackwell) et AMD (RDNA3, RDNA4) : trois crans de précision pour l'agrandissement (SPAN) et l'interpolation (RIFE, DistilDRBA, GMFSS) :
**Performance FP8 (E4M3)**, **Équilibre FP16**, **Qualité FP32**. Chaque cran a SON graphe, exporté pour lui (pas un graphe FP16 converti à la volée), avec la fusion complète que TensorRT fait déjà en FP16
(par exemple conv 1×1 d'attention + Tanh + Add + Mul en une couche) étendue au FP8 : flux complet en E4M3, échelles statiques, activations qui restent en FP8 d'une couche à l'autre.
**Règle de décision :** une piste n'entre dans le lecteur que si elle gagne **au moins 8 % en ABAB** (quatre tours alternés, même séance, `--useCudaGraph`, 300 itérations) **et** garde la qualité (PSNR/SSIM contre l'existant sur de vraies images d'anime) ; sinon elle reste dans `tools/bench/fp8/`.

### Mesures du 28/10 (RTX 4070 Ti, TensorRT 11.3.0.99, SPAN Balanced HD V3.1, 1088×1920)
| Variante | ms (médiane GPU) | Remarque |
|---|---|---|
| FP16 (référence) | 15,4 à 15,9 | varie d'une séance à l'autre : comparer dans la même séance |
| FP8 Q/DQ partout, poids par tenseur | 18,3 à 19,0 | +17 % plus lent : passes Q/DQ/Cast séparées (8 à 10 ms) |
| FP8 Q aussi sur la sortie de la conv d'attention | 18,1 à 18,2 | −4 % contre le FP8 de base |
| FP8 mixte (c1/c2/c3 seuls) | 17,6 à 18,8 | plus lent que le FP16 ; qualité 53,7 dB / SSIM 0,998 contre FP16 |
| SPAN sans attention, FP16 / FP8 | 12,65–13,38 / 13,54–13,60 | les convs FP8 gagnent 27 % (8,60 contre 11,86 ms) mais 6,2 ms de passes Add/Q restent |
| Motif « Conv → Q → DQ → Add(saut) → Q » (chaîne résiduelle 544×960) | −17 % contre FP16 | le motif qui fusionne |
| Microbanc convolutions 3×3, 8 couches | 64 canaux −24 %, 128 −14 %, 256 −13 %, 48 canaux +5 % | le FP8 n'aide que ≥ 64 canaux |
| RIFE v4.25 768×1280, FP16 / FP8 | 8,90–8,98 / 8,33–8,47 | −6 % ; PSNR 34 à 57 dB contre FP16 |
| GMFSS 1280×768 (FP16) | pair 45,4 ; demi-flot 8,6 ; fusion 11,2 | FP8 non essayé (entrées multiples) |
| Tiling TensorRT 1 et 3 (FP16) | 15,80 / 15,61 contre 15,37 | aucun gain |

**Causes.** Les convolutions sont limitées par le calcul (~91 TFLOPS en FP16, près du plafond) : TensorRT les fait déjà à ~90 % du pic ; l'attention et les jonctions sont limitées par la mémoire (504 Go/s). En FP8, TensorRT ne fond pas la re-quantification dans les noyaux d'attention (couches 41 → 58, passes `TanhAddMulMulCast` ~0,7 ms × 8 et `MulCast` seuls ~4 ms).
**Pièges.** TensorRT 11 n'a plus `--fp16/--fp8` (réseau fortement typé) ; une calibration avec une entrée mal formée donne des échelles dégénérées et des vitesses FAUSSES (vérifier toujours le PSNR) ; ONNX Runtime CPU refuse un Transpose en FP8.

### Chantiers (ordre)
1. **Plugin TensorRT fusionné SPAN (FP8)** : noyau « Add(saut) + Tanh/Mul + requantification E4M3 », un bloc d'abord, plafond 2 h de prototype (branche `claude/fp8-kernels`, agent isolé). Gain attendu −10 à −15 %.
2. **Graphes spécialisés sans noyau maison** : placement des Q/DQ (motif qadd), précision mixte (première, dernière conv et attention en FP16), calibration sur images d'anime réelles (percentile ou entropie), échelle par canal des poids si le noyau la fond, **ModelOpt** (`--quantize_mode fp8`, `--autotune`) comparé à nos graphes manuels.
3. **Équilibre FP16 et Qualité FP32 spécialisés** : suppression des Reformat/Cast/unshuffle superflus (E/S NHWC `hwc8`), canaux multiples de 32, activations plus fusionnables (hard-tanh ou polynôme à la place de Tanh), précision mixte (îlots FP32 seulement où nécessaire), tactiques par réseau, `maxAuxStreams` sur RIFE ; mesure ABAB et PSNR contre l'existant.
4. **Prototype CUTLASS** : conv 3×3 64→64 FP16 puis FP8 sm89 avec épilogue biais + Add(saut) + Tanh/Mul, comparée à TensorRT ; seule voie vers −15 à −25 % ; demande l'accord de Tatoub pour la dépendance (BSD-3, header-only) : l'accord a été donné pour le prototype, pas pour l'intégration.
5. **Interpolation** : RIFE (convs larges, déjà −6 % en FP8 TensorRT seul, −20 % non confirmé : la mesure initiale était fausse), DistilDRBA et GMFSS fusion (surtout élémentaire : 89 à 96 % du trafic convertible) avec le même convertisseur ; GMFSS pair garde warp/softmax/corrélation en FP16.
6. **QAT** (affinage quantification-aware par distillation FP16 → FP8, +5 à 10 dB typiques) : seulement si les poids PyTorch d'origine sont disponibles (non trouvés sur le PC de test).
7. **AMD** (fil séparé, aucune carte sur le PC de test) : Vulkan `VK_KHR_cooperative_matrix` FP8 (`GL_EXT_float_e4m3`, compile avec glslang 16.6) et MIGraphX ; fusion de l'épilogue du noyau de convolution (biais, activation, Add de saut) puis du bloc c1-c2-c3 ; RDNA4 d'abord pour le FP8 matériel.
8. **Pipeline** (mesurer avant de toucher) : recouvrements décodage/réseau/rendu, copies, mémoire épinglée ; CUDA graphs, mémoire épinglée et files séparées existent déjà.

## Suivi du 28/10, 15:20 (ajouts, rien retiré)
- FAIT depuis la dernière mise à jour : #82 HRIR, #84 poses par défaut, #85 règle getenv (AppImage), #86 coopmat RDNA 4, #87 menu (molette, 5 catégories), #89 barre fine, #92 section précision.
- En revue : #88 (crans de précision), #90 (tuiles FP8 Vulkan), #91 (Q/DQ FP8 dans gpu_ir).
- Release v0.5.1 publiée (Linux + Windows, pré-release, manifeste preview) ; Tatoub confirme TensorRT fonctionnel sur son Fedora.
- CUTLASS (BSD-3) cloné et ModelOpt 0.47.0 (Apache-2.0) installé hors du dépôt sur le PC de test (accord direct de Tatoub, 15:11) ; décision du coordinateur : pas de QAT ni de poids PyTorch à télécharger.
- Reste avant la v0.6 : mesures GPU en file (plugin SPAN, SPAN mixte a/b, ModelOpt FP8, prototype CUTLASS, GMFSS et DistilDRBA FP8, E/S NHWC), reconstruction du paquet TensorRT (empreinte #67/#70), test de lecture complet sur main, libellés et disposition horizontale du menu, assistant guidé du moteur, Auto en VRR (demi-paliers), archive TensorRT Windows, notes et version 0.6.0 ; étiquette seulement sur le message direct de Tatoub.

### Mesures complémentaires du 28/10 (après-midi, mêmes conditions : 4070 Ti, TensorRT 11.3, ABAB, vraie calibration)
| Variante SPAN Balanced 1088×1920 | FP16 de la séance | variante | écart |
|---|---|---|---|
| FP8 mixte (a) : convs c1/c2/c3 seules | 14,2 à 15,9 ms | 17,3 à 18,5 ms | +13 à +25 % (plus lent) ; qualité PSNR 53,7 dB, SSIM 0,998 |
| FP8 mixte (b) : (a) + Q sur la sortie de c3 | idem | 17,97 à 20,06 ms | +13 à +30 % |
| ModelOpt 0.47.0 `--quantize_mode fp8` (calibration max, 1 image) | 15,66 à 15,82 ms | 16,57 à 17,21 ms | +6 à +10 % ; PSNR 46,9 dB (pire 40,4), SSIM 0,981 |
| Plugin TensorRT fusionné « Add+Tanh/Mul+Q » (agent, blocs 2 à 7) | 16,28 à 16,42 ms | 20,60 à 20,63 ms | +25 % : le plugin coupe les fusions Myelin voisines (1,3 à 1,8 ms de reformatages et copies par bloc) ; noyau seul 0,31 ms contre 0,43 |
| FP16 : `--tilingOptimizationLevel` 1 / 3, `--persistentCacheRatio` 0,5 | 15,4 à 15,8 ms | 14,7 à 16,3 ms | aucun gain stable |
| RIFE FP16 : `--maxAuxStreams` 4 / 0 | 8,4 à 10,0 ms | 7,8 / 8,1 ms | ≈ −3 à −5 % pour 4 files (mesure à reconfirmer) |

**Vulkan, profil par dispatch (PR #100), RIFE v4.25 768×1280 FP16, noyaux par défaut** : 135 dispatches, 14,25 ms ; conv coopmat 53 dispatches 8,23 ms (57,7 %) ; `net_copy_f16` 37 dispatches 2,40 ms (16,8 %, 68 % des octets écrits) ; grid_sample 1,02 ms ; binary 1,00 ms ; resize 0,85 ms.

**Bilan.** Aucune variante FP8 sous TensorRT 11.3 ne bat le FP16 sur SPAN (meilleure : ModelOpt, +6 %) ; RIFE gagne −6 % seulement. « Performance FP8 » reste désactivé par défaut pour SPAN tant qu'aucun gain n'est mesuré. Reste ouvert : prototype CUTLASS (convolution maison à épilogue fusionné, hors TensorRT), GMFSS/DistilDRBA FP8, E/S NHWC.


- Son anime (audio) : module dédié aux pistes stéréo sur sortie stéréo (spatialisation milieu/côtés, voix, graves harmoniques, sonie BS.1770, compresseur, limiteur vrai-crête), niveaux Naturel / Anime / Anime fort, par appareil, bouton Comparer ; mesures et limites dans `docs/audio-anime.md`, rendu à valider à l'oreille par Tatoub.


19. Profils audio audibles sur du stéréo : FAIT (#110) : Voix claires et Nuit ont un égaliseur, note dans le panneau, test chiffré ; cause : en « Enceintes » les profils surround n’agissaient pas sur une piste stéréo.
20. « Son anime » stéréo : FAIT (#114) : spatialisation (enceintes et casque), voix, sonie EBU R128, limiteur vrai-crête, graves, trois niveaux, par appareil, comparaison ; mesuré hors ligne, **à écouter** (voir `audio-anime.md`).
21. Pop-up « lecteur par défaut » : petite fenêtre déplaçable, indépendante du panneau : FAIT (#111) ; **à vérifier à l’écran panneau fermé**.
22. Fusion pleine résolution (TensorRT FP16 + noyaux maison) : RIFE FAIT (#104, 1,21 ms contre 3,76 ms par paire), DistilDRBA FAIT (#107, 0,37 ms contre 0,74 ms), GMFSS livré **éteint** (gain −6 à −9 % sur la chaîne, −15 % au mieux sur la paire ; pas de FP8 GMFSS) : voir `rife-fusionne.md`, `drba-fusionne.md`, `gmfss-fusionne.md`.
23. SPAN FP8 par noyaux CUTLASS (cran Performance) : FAIT (#103 : 9,1–9,3 ms contre 14,25 ms, 50–52 dB) et paquet `.span8` dérivé par le lecteur (#106) ; FP8 RIFE/DistilDRBA sous TensorRT : non retenu (−6 % ou négatif).
24. Tour de lecture en 720p (4070 Ti, ×2,5, facteur fixe, verrou GPU) : RIFE et DistilDRBA fusionnés tiennent ≈ 56 i/s sans repli ; GMFSS ne tient pas ×2,5 à 720p (41 i/s). Les essais antérieurs (540p, Auto actif, chevauchements, stream en fond) sont écartés.
25. FP8 des réseaux d'interpolation (`docs/fp8-interpolation.md`) : GMFSS paire FP8 hors du squelette de GMFlow FAIT, **par défaut** (−15 % en ABAB, 46,9 dB base / 48,6 dB union au pire cas, SSIM ≥ 0,998) ; DistilDRBA entier −11 % qualité tenue mais coupe le fusionné (2× plus rapide) : fichier facultatif, éteint ; RIFE : −5 % (4.25) ou +27 % (4.22 lite), pire cas 31 dB, sous-moteurs +16 % : aucun graphe publié. Graphes à publier par `tools/release/publier-modeles-fp8.sh` (Tatoub).


25. FP8 : GMFSS livré en FP8 (cran Performance par défaut, −15 % sur la paire, #125, graphes publiés dans models-interp-1) ; RIFE et DistilDRBA : aucun graphe FP8 (plus lents ou hors garde-fous : docs/fp8-interpolation.md) ; SPAN : cran Performance sélectionnable, paquet .span8 fabriqué automatiquement au premier démarrage avec ce cran (#133, en revue).
26. Mode Auto : « Stable » (par défaut : prédiction du coût de service, remontée rapide, mémoire par vidéo, #121, #123) et « Rapide (expérimental) » (instants exacts calés sur l'horloge, VRR ou vsync, tous moteurs sauf Apple, #124, #130), plafond optionnel Maximum / 120 / 60 i/s (#122). Mesure sur la scène lourde : Stable 43 à 51 i/s en moyenne avec 0,15 % de pertes ; Rapide 61 à 67 i/s avec 0,05 à 0,18 %.
27. Décision de Tatoub (29/10, 12:23) : plus aucune nouvelle fonctionnalité ; objectif 1.0 STABLE, parfaite au minimum sur NVIDIA Linux et Windows. Abandonné : modèle SPAN AniSD pour le SD (#132 fermée), précalibrage du SPAN FP8. En cours : audit esthétique de l'interface (audit-ui) et audit fonctionnel de chaque fonction (audit-fonctions : docs/audit-fonctions.md).