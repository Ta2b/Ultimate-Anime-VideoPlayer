# Plan de la v0.6

Les retours de Tatoub du 28 octobre (après la v0.5.0, publiée en pré-version) forment la **v0.6**, pas une v0.5.1. La version CMake passe à 0.6.0 **à la
préparation de la release v0.6**, pas avant. Même règle qu'en v0.5 : une PR par étape, base main, clang-format avant chaque PR ; l'étiquette n'est posée
que sur un accord clair de Tatoub dans le fil. Détail des décisions et journal : ce fichier ; l'historique de la 0.5 : `plan-v0.5.md`.

## Étapes, dans l'ordre (état au 28/10, 15:20)
- [x] Catalogue RIFE réduit à 4.25 et 4.22 lite (#48) ; menu SPAN réduit au catalogue AnimeJaNai HD V3.1 (#55) : plus de profil « legacy » dans les listes
- [x] Fenêtre fantôme à la fermeture (Fedora, Wayland, GNOME) : correctif défendable par la lecture du code (#49) ; **à confirmer sur le Fedora GNOME**
- [x] Interpolation Auto par défaut pour tous les profils (#53)
- [ ] « Garder les poses DRBA » par défaut : **performance mesurée** (poses actives : tient, retard 0,0 ms, composition 1,2 ms ; inactives : ne tient pas, 15,1 ms, 6,0 ms ; 4070 Ti, Re:Zero 18, ×3) ; qualité : comparaison d'images en cours, puis décision
- [x] HDR : écran SDR sans conversion HDR (options grisées, note), mode SDR gradé pour imiter le rendu HDR, RTX HDR non installé : repli sur l'adaptatif (#68, #69) ; **à confirmer chez Tatoub**
- [x] Interface : style Showtime (commandes rondes au centre, titre en gras), fenêtre « lecteur par défaut » au démarrage, réglages avancés dans les onglets existants sans catégorie Avancé, Statistiques et Réinitialiser dans Général (#71, #72) ; reste le menu : molette, barre de défilement, libellés, disposition horizontale
- [ ] Benchmark autonome et silencieux : séquence de test écrite par le lecteur, fenêtre minuscule, étapes annoncées (#75) ; **reste** l'exploration des moteurs, multiplicateurs et résolutions, Auto en dernier
- [x] Moteurs : espace disque vérifié avant l'installation, dossier d'installation au choix (#54) ; installation TensorRT corrigée : architecture du GPU, progression unique, un seul essai (#60) ; paquet TensorRT découplé des étiquettes (#67)
- [x] Audio : profils de virtualisation (#64), distance, pièce, égaliseur, nuit, profil par appareil (#80), **à écouter** ; [ ] convolution HRIR à la HeSuVi (#82, en revue)
- [x] Empaquetage : AppImage (#74, essai local #77), installeur .exe Inno Setup + zip portable (#74, **à valider par la prochaine release Windows**), .dmg (#79)
- [x] Police : repli CJK vérifié (#59), chargeur FreeType pour le CFF2 de Fedora (#70) ; **à confirmer chez Tatoub**
- [x] CI allégée (#73) ; dépôt des releases séparé **gardé** (décision de Tatoub, 28/10)
- [ ] Release v0.6 : version 0.6.0, notes `tools/release/notes/v0.6.0.md` (brouillon #81), liste des essais Windows/macOS ; étiquette seulement sur le message direct de Tatoub

## Journal
- **2026-10-28** : v0.6 ouverte. Mesures du contrôleur Auto sur la tête 00f00ee : RIFE ×4, DistilDRBA ×5, GMFSS ×2 tenus en fin de run (`docs/reports/mesures-auto-4070ti-2026-10-28.md`).
- [x] Paquet TensorRT découplé des étiquettes : empreinte de ses entrées (`trt-fingerprint.sh`), release fixe `trt-<empreinte>` (`trt-package.sh`), Release Linux la lit sans relance ; plus de reconstruction ni de seconde passe par release (demande de Tatoub, 0.5.1)

## Liste vivante du détail demandé par Tatoub (28/10) : état

1. Audio : FAIT (#64, #80) : profils, distance, pièce, égaliseur, nuit, profil par appareil ; à écouter.
2. Réglages avancés : FAIT (#61, #72) : dans les onglets existants, OLED automatique, Réinitialiser et Statistiques hors des avancés.
3. Pop-up « lecteur par défaut » et réglage discret : FAIT (#71) ; composition finale et soins OLED hors de Couleurs : FAIT (#72, dans Agrandissement et HDR avec les avancés).
4. HDR : FAIT (#68, #69) ; à confirmer chez Tatoub.
5. Benchmark autonome et silencieux : EN PARTIE (#75) ; reste l'exploration des moteurs, multiplicateurs et résolutions.
6. Poses DRBA par défaut : EN COURS : performance mesurée (favorable), qualité à comparer.
7. Design Showtime FAIT (#71) ; AppImage FAIT (#74, #77) ; installeur .exe + zip portable FAIT (#74), à valider par la release Windows ; .dmg FAIT (#79).
8. Fedora GNOME : fenêtre fantôme (#49) et japonais (#59, #70) : À CONFIRMER chez Tatoub.
9. Convolution HRIR à la HeSuVi : EN REVUE (#82) ; préréglage « Voix d'anime » FAIT (#80).
10. Auto en VRR, demi-paliers ×0,5 : À FAIRE (trancher, mesurer, répondre).
11. Menu : molette, barre de défilement, moins de catégories, disposition horizontale, libellés français : À FAIRE (pastille « Le plus lent » FAITE).
12. Arbre : EN PARTIE (#72) ; à revoir : place de Langues, Export, Moteurs, Couleurs sans réglages sans rapport.
13. « Ouvrir avec » enregistré au premier lancement : FAIT (assistant).
14. Config auto (tranche du GPU et résolution de l'écran, réduire puis agrandir, SPAN Sharp1 Performance, modèle par tranche) : FAIT (#53, tests `[auto_config]`).
15. Assistant : choix guidé du moteur : À FAIRE.
16. Poses DRBA : contradiction du tableau de décisions (« désactivées ») : à corriger après la décision (point 6).
17. .dmg FAIT (#79) ; archive TensorRT Windows manquante : À FAIRE (même schéma que Linux, #67) ; build macOS corrigé (#66).
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

