# Rapport de l'audit 41. HDR, grade et looks

Branche : `claude/project-thread-wda93a`. Commit audité : `f7e00ae`. Date : `2026-11-09`.

## Verdict

Le code du grade est propre et bien borné, mais ce qui sort à l'écran n'est pas ce que ses parties annoncent : la
pile de LUT et la compression finale placent le blanc des surfaces vers 70 % du pic de l'écran et divisent par deux
la montée des émetteurs, la table composée en 65³ s'écarte de la pile exacte jusqu'à 18 codes PQ 10 bits à l'écran,
et le limiteur de flashs fait sauter l'image de 695 à 228 nits sans transition.

## Cadrage retenu

Aucune question posée à Tatoub, sur consigne du coordinateur : tout est audité au maximum.

- Demande de Tatoub (09/11) : la qualité du rendu ne doit être négligée nulle part ; une baisse de qualité faite
  pour la vitesse est un défaut, sauf mesure qui la justifie.
- Cible : RTX sous TensorRT, Windows 11 et CachyOS (KDE). macOS est hors cible, le VRR hors sujet.
- La zone ne contient ni RIFE ni DistilDRBA. Le seul lien est le renvoi sur les statistiques du grade, vérifié pour
  les deux chemins (voir « Défauts connus revérifiés »). Rien dans cette zone ne freine DistilDRBA.
- Les 29 fichiers de la fiche ont été lus en entier, ainsi que `renderer.c`, `custom_mpv.c`, `gpu.h` et
  `vulkan/swapchain.c` de libplacebo v7.360.1 (la version épinglée par `tools/deps/linux-from-source.sh:37`), pour
  ne rien citer de mémoire.
- Le fil n'a ni GPU ni dépendances construites : aucun test GPU n'a tourné. Les mesures ci-dessous viennent du code
  CPU du dépôt. Ce qui demande l'écran ou la carte est marqué « à confirmer ».

## Mesures faites ici

Un programme hors dépôt compile `src/render/cube_lut.cpp` tel quel, charge les quatre fichiers de `data/luts/`
comme le fait `load_look` (avec `fill_unreachable`), monte la pile par défaut de `anime_look` (11 lectures) et
rejoue en double précision la passe « kwin » de `anime-hdr-black.hook:136-185`. Modèle de l'entrée : source SDR en
gamma 2,4 pur, blanc à 203 nits, relèvement fixe de 1,15, BT.709 porté en BT.2020 puis en PQ ; blanc SDR de KDE à
203 nits (le cas de Windows). Limites : les passes 1 à 3 du grade ne sont pas simulées au-delà du relèvement, et
libplacebo applique BT.1886 avec compensation du point noir, ce qui déplace les ombres d'environ 15 % (calcul à la
main, non mesuré).

**Courbe de sortie sur l'axe neutre, écran de 1 000 nits**

| Code RVB 8 bits | Entrée de la pile (nits) | Sortie de la pile (nits) | À l'écran (nits) | À l'écran, limiteur de flashs (pic 250) | À l'écran sans pile |
|---|---|---|---|---|---|
| 16 | 0,30 | 0,02 | 0,02 | 0,02 | 0,30 |
| 32 | 1,60 | 0,65 | 0,64 | 0,64 | 1,59 |
| 48 | 4,24 | 3,95 | 3,88 | 3,88 | 4,16 |
| 96 | 22,4 | 24,8 | 22,3 | 22,1 | 20,3 |
| 128 | 44,6 | 54,9 | 44,2 | 43,3 | 37,3 |
| 160 | 76,3 | 193 | 107 | 99 | 57 |
| 192 | 118 | 1 783 | 313 | 190 | 78 |
| 224 | 171 | 4 505 | 547 | 217 | 99 |
| 255 | 233 | 6 295 | 695 | 228 | 119 |

**Au-dessus du blanc (ce que le grade donne aux émetteurs), écran de 1 000 nits**

| Niveau avant la pile (nits) | 233 (surface blanche) | 337 (émetteur, Éclats à 100 %) | 500 | 775 (Éclats à 200 %) | 1 000 |
|---|---|---|---|---|---|
| À l'écran (nits) | 695 | 826 | 900 | 960 | 974 |

**Blanc des surfaces selon le pic de l'écran**

| Pic de l'écran (nits) | 400 | 600 | 800 | 1 000 | 1 500 | 2 000 |
|---|---|---|---|---|---|---|
| Blanc des surfaces (nits) | 322 | 446 | 571 | 695 | 1 006 | 1 317 |
| Gris à 18 % (nits) | 40,8 | 41,0 | 41,2 | 41,4 | 41,9 | 42,4 |

**Table composée contre la pile exacte**, 400 000 couleurs SDR (un quart près des gris), en codes PQ 10 bits

| Table | Sortie de la pile : max, moyenne, part à plus d'un code | À l'écran (après la passe kwin) : max, moyenne, part à plus d'un code |
|---|---|---|
| 65³ trilinéaire (le code actuel) | 15,4 ; 0,90 ; 29,3 % | 18,0 ; 0,65 ; 19,4 % |
| 129³ trilinéaire | 8,8 ; 0,27 ; 4,8 % | 10,3 ; 0,21 ; 2,8 % |
| 65³ tétraédrique | 17,5 ; 1,10 ; 34,6 % | 12,6 ; 0,76 ; 25,8 % |

Autres relevés : composer la pile coûte 163 ms en 65³ et 1 243 ms en 129³ (un cœur du conteneur, `-O2`) ; le texte
de la texture 65³ pèse 4,46 Mo et prend 16,5 ms à écrire (il pèserait 35 Mo en 129³) ; l'arrondi de la table en
demi-flottant ajoute au plus 0,25 code ; la courbe neutre est monotone (baisses de 0,016 code au plus, sous
0,01 nit) ; aucun canal n'atteint l'écrêtage de la passe kwin (au plus 0,77 fois le pic) ; le gain de contraste
local de la pile et de la passe kwin (d log sortie / d log entrée) vaut 2,0 sous 2 nits, 0,9 à 1,0 entre 10 et
40 nits, 2,7 entre 80 et 100 nits, 0,6 au blanc.

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A41-01 | Majeur | `src/render/cube_lut.cpp:287-314`, `src/render/cube_lut.h:19`, `:38-42` ; `src/app/play/look_catalog.cpp:83-131` | Doublon d'A49-05, confirmé et précisé. La pile par défaut (11 lectures de LUT de 33 points) est réduite à une table de 65³. Reproduit : 15,4 codes PQ 10 bits au pire à la sortie de la pile, 0,90 en moyenne, 29,3 % des couleurs à plus d'un code. Ce que l'audit 49 ne disait pas : à l'écran, après la passe kwin, l'écart est de 18,0 codes au pire, 0,65 en moyenne, 19,4 % à plus d'un code ; passer à 129³ ne le supprime pas (10,3 codes au pire, 2,8 % à plus d'un code) et le tétraédrique en 65³ est pire en moyenne. L'écart vient des cassures de pente des LUT de 33 points, que les LUT précédentes déplacent hors de la grille : seule la chaîne est exacte. Le gain de vitesse de la composition n'est mesuré nulle part. | Toute lecture en HDR avec le grade : les teintes saturées (pire cas mesuré : RVB SDR 0,18 ; 0,34 ; 0,95, un bleu) sortent décalées de plusieurs codes, de façon non uniforme. | Enchaîner dans le shader les quatre LUT de 33³ (11 lectures, 22 avec `look_clean`), mesurer le coût de la passe, et ne garder une table composée que si ce coût fait perdre la cadence. À défaut, 129³ créée par `pl_tex_create` (35 Mo de texte par le hook sont à éviter). Corriger le commentaire « sans écart ». | Mesuré (tableau ci-dessus). À confirmer sur la carte : temps de la passe du look (`grade_gpu_us`) en table 65³, 129³ et en chaîne, à 1440p et 165 Hz. |
| A41-02 | Majeur | `src/render/shaders/anime-hdr-anime/3-anime.hook:119`, `:124-129` ; `src/render/shaders/anime-hdr-black.hook:136-185` ; `src/render/grade_params.cpp:128-146` ; `src/app/play/placebo_grade.cpp:226-232` | Ce que le grade décide est défait par la pile et la passe kwin, qui viennent après lui. (1) Le blanc des surfaces sort à 695 nits sur un écran de 1 000 nits, et vers 70 % du pic sur tout écran (tableau), alors que le code et la doc parlent d'un blanc de source à 203 nits et de surfaces « ancrées ». (2) Un émetteur blanc, que le grade monte de 0,53 diaphragme au-dessus d'une surface blanche, n'en garde que 0,25 à l'écran (826 contre 695 nits) ; à Éclats 200 %, 0,47 (960 nits). (3) Le « limiteur de l'OLED » (marge des éclats divisée par deux en scène claire, 3-anime.hook:119) ne change presque rien : 500 nits avant la pile donnent 900 à l'écran, 1 000 donnent 974. (4) Le MaxFALL annoncé sans contrat d'écran vaut 233 nits (`sdr_white × 1,15`) alors qu'une image blanche fait 695 nits. | Scène claire plein écran sur un OLED de 1 000 nits : les surfaces demandent 695 nits là où la dalle en tient environ 260 en plein champ (`display_model.cpp:16`). C'est le limiteur de la dalle qui assombrit l'image, pas celui du grade. | À décider par Tatoub (question 1). Si le rendu actuel est le bon, écrire ces niveaux dans `docs/hdr-anime.md`, faire agir le limiteur et le curseur Éclats après la pile (dans la passe kwin, par le pic visé), et annoncer le vrai MaxFALL. Sinon, recaler la pile. | Mesuré sur le modèle. À confirmer : instantané `.pq.ppm` d'une mire blanche et d'une mire à émetteur, lu en nits ; ou sonde sur l'écran. |
| A41-03 | Mineur | `data/luts/HDR grading base ver.9.0 33 point.cube` (nœuds 1 à 5 de l'axe neutre) ; `docs/hdr-anime.md:63-67` | La LUT de base écrase les ombres : 0,30 nit en entrée (RVB 16 sur 255) sort à 0,02 nit, 1,6 nit (RVB 32) à 0,65. Tout ce qui est sous RVB 16 est noir. La plage de 0 à 1 nit n'est décrite que par cinq nœuds de la LUT. La doc dit pourtant vouloir « garder lisibles les détails des nuits », et le curseur « Noirs profonds » creuse encore par-dessus. | Plan de nuit dont les détails vivent entre RVB 10 et 30 : ils disparaissent sur un OLED. | À décider à l'œil (question 2). Si c'est voulu, l'écrire ; sinon adoucir le pied de la LUT de base ou la remplacer sous 2 nits. | Mesuré sur le modèle (gamma 2,4 pur ; l'écart de BT.1886 ne change pas l'ordre de grandeur). À confirmer : trois plans de nuit, avec et sans grade. |
| A41-04 | Mineur | `src/app/play/placebo_grade.cpp:206-225`, `src/render/grade.cpp:418-420`, `src/app/play/chain_hdr.cpp:108-123` | Sans pile (fichier `.cube` manquant ou illisible, ou hook refusé), la passe kwin reste active : elle compresse une image qui n'a pas été montée par la pile, et le blanc tombe à 119 nits au lieu de 695. L'échec du hook composé (`adopt`, `!ready->hook`) n'écrit aucune ligne de journal. | Un fichier du même nom, abîmé, dans le dossier `luts` de l'utilisateur : « Look unreadable » s'affiche une fois, puis toute la lecture est sombre. | Couper la passe kwin (ou la rendre neutre) quand aucune pile n'est composée, et journaliser l'échec dans `adopt`. | Mesuré sur le modèle (colonne « sans pile »). |
| A41-05 | Mineur | `src/render/grade.cpp:361-365`, `src/app/play/chain_hdr.cpp:125-135`, `:218-223` | `set_looks` valide les LUT en lisant un hook complet sur le thread de la boucle, puis le détruit : 4,46 Mo de texte écrits (16,5 ms ici), décodés par libplacebo, et une texture 65³ envoyée au GPU pour rien. Le travail de fond de P4.9 refait ensuite la même chose. La validation est inutile : `parse_cube` a déjà vérifié taille et valeurs. | Peu après le début de chaque lecture (quand `LookLoader` rend la pile), et à chaque cran des curseurs de contraste ou d'éclats du look : plusieurs images perdues à 165 Hz. | Retirer la lecture d'essai de `set_looks` (ou la faire dans le thread des LUT). Passer la pile déjà composée par `anime_look` sans la recomposer. | Mesuré pour l'écriture du texte. À confirmer : trace `AJP_TRACE` autour de `show_looks` sur la carte. |
| A41-06 | Mineur | `src/render/shaders/sharpen.hook:14`, `:27-31` ; `src/render/sharpen.h:1`, `src/render/shaders.h:21`, `src/app/play/placebo_screen.cpp:155` | La netteté est posée sur `SCALED`, alors que trois commentaires disent `MAIN`. À cet étage, l'image est en gamma quand aucune mise à l'échelle n'a lieu, et en lumière linéaire dès que libplacebo agrandit ou réduit (`renderer.c:1993-2079` : sigmoïde ou linéaire, non refermés avant `SCALED`). Les seuils du shader (1 % et 4 % d'écart, marge relative) ne valent que pour un domaine. Elle s'applique aussi après le filtre final, donc sur l'image déjà agrandie d'un écran 4K. | Netteté à 50, plein écran 1440p (anneau à la taille de l'écran) puis fenêtre réduite : la force change, et en linéaire les ombres ne sont plus accentuées. Sans effet par défaut (netteté à 0). | La poser avant le filtre final dans un domaine fixe, ou ramener l'image en gamma dans le shader. Corriger les commentaires. | Lu (dépôt et libplacebo). À confirmer : deux instantanés, plein écran et fenêtre réduite. |
| A41-07 | Mineur | `src/render/grade_params.cpp:158-166`, `src/app/play/chain_hdr.cpp:345-384`, `src/app/play/placebo_grade.cpp:348` | Sur une sortie SDR, le vrai noir OLED n'est jamais appliqué (`sdr_tonemap` remet ses seuils à 0), mais il est mesuré et annoncé : « True black: 0 below … nit ». La mesure convertit en plus les codes SDR avec le pic HDR (`params.peak`, 1 000) au lieu du blanc SDR. | OLED avec le HDR coupé dans le système, vrai noir sur Auto : le panneau dit que le vrai noir est actif, l'image ne change pas. | Appliquer le pied dans la dernière passe SDR, ou afficher « sans effet sur cette sortie ». | Lu. |
| A41-08 | Mineur | `src/render/grade_params.cpp:115-116`, `src/app/play/placebo_grade.cpp:214`, `:236-243` | Une source déjà HDR n'est pas gradée (défaut), et dans ce cas rien n'est annoncé à la swapchain : libplacebo croit la cible à 10 000 nits et ne ramène pas la source au pic de l'écran, aucune métadonnée ne part. | Fichier HDR10 masterisé à 4 000 nits sur un écran de 1 000 nits sous Windows : les hautes lumières sont écrêtées par l'écran. | Annoncer le pic de l'écran même sans grade, pour que libplacebo fasse sa compression. | Lu. À confirmer sur l'écran avec un fichier HDR10. |
| A41-09 | Mineur | `src/render/grade.cpp:275-298`, `:316` | Thread des LUT sans `try` (GPU-11, A30-04, renvoi 30 vers 41) : confirmé. Le corps alloue 3,3 Mo de table et 4,5 Mo de texte ; un `bad_alloc` mène à `std::terminate`. | Mémoire courte au changement d'un curseur du look. | Entourer le corps de `run` d'un `try` qui rend un `Refusal` dans `done`. | Lu. |
| A41-10 | Mineur | `src/render/grade_params.cpp:56-70`, `src/app/play/placebo_grade.cpp:57-65`, `src/render/grade.cpp:57-69`, `src/render/emitters.cpp`, `src/render/faces.cpp`, en-têtes de `anime-hdr-black.hook:1` et `anime-hdr-look.hook:100` | Dette. (1) `GradeMode::Sdr` et toutes les branches `sdr_out > 0.5` des shaders ne sont jamais jouées : dès qu'une source SDR est gradée, `sdr_anime` prend le pas sur `sdr`. (2) Quinze emplacements de LUT pour une seule pile : quinze textures créées à chaque lecture du hook, quatorze neutres. (3) Le chemin des LUT SDR (`look_white`, `look_gamma`) n'est plus atteint. (4) `emitters.cpp` et `faces.cpp` ne servent qu'aux tests et aux outils. (5) En-têtes de shaders périmés (« dixième passe », « grade fusionné », marques `//@…_END@`). (6) La pile est composée deux fois (`anime_look`, puis `Grade::Impl::compose`). | Aucun effet à l'écran ; le test « sortie SDR : le vrai noir OLED » garde vivant un chemin mort. | Retirer après décision sur A41-07. | Lu. |
| A41-11 | Mineur | `tests/unit/render/cube_lut_test.cpp:146`, `tests/unit/render/grade_gpu_anime_test.cpp:77`, `tests/unit/app/look_v2_test.cpp:34` | Aucun test ne passe par les quatre fichiers livrés, ni par la chaîne grade, pile et passe kwin. Les tests « HDR anime » jugent la passe 3 seule, dont l'effet est divisé par deux à l'écran. Le test « sans écart » ne couvre qu'une LUT. | Une retouche d'un `.cube` ou de la composition change le rendu sans qu'un test bouge. | Trois tests sans GPU, qui échouent aujourd'hui : (1) pile par défaut composée contre chaîne exacte, écart maximal sous 2 codes ; (2) niveau du blanc et du gris à 18 % de la courbe neutre dans une plage écrite ; (3) RVB 16 au-dessus d'un plancher en nits, si A41-03 est retenu. | Lu ; les tests sont décrits, non écrits (un seul fichier autorisé). |

## Défauts connus revérifiés

| ID | Statut | Note |
|---|---|---|
| GPU-07 | Toujours présent, aggravé | `chain_hdr.cpp:252` plafonne le pic à 250 nits d'un coup, `placebo_grade.cpp:217-243` réannonce alors le maximum (1 000 puis 250). `vk_sw_colorspace_hint` ne recrée pas la swapchain, mais l'écran reçoit de nouvelles métadonnées. S'y ajoute un saut d'image sans transition : le blanc passe de 695 à 228 nits à l'entrée du limiteur, et remonte de même à la sortie (`chain_hdr.cpp:321-327`). Correction proposée : faire glisser le pic sur une demi-seconde et laisser les métadonnées au pic de l'écran. |
| GPU-08 | Toujours présent | macOS : `options.h:125-129`, passe kwin éteinte (hors cible). Wayland hors KDE : `placebo_grade.cpp:193-195` suppose 203 nits. À confirmer sous GNOME avec son réglage de luminosité HDR. Sous Windows, le choix de 203 nits est correct. |
| GPU-11 | En partie réfuté | Créer une texture depuis un second thread est permis : `pl_gpu` est sûr entre threads sous Vulkan (`limits.thread_safe`, `vulkan/gpu.c:527`). Reste l'absence de `try` (A41-09). |
| GPU-12 | Toujours présent | `chain_hdr.cpp:153-163` ; traité par A30-02, rien à ajouter. |
| GPU-14 | Toujours présent | `chain_hdr.cpp:367`. |
| GPU-15 (fiche, banding) | Précisé | La pile étire le contraste local jusqu'à 2,7 fois (80 à 100 nits) et 2 fois sous 2 nits. `look_clean`, toujours actif (`chain_hdr.cpp:94`), garde leur contraste d'origine aux écarts de moins de 6 % : les paliers d'un anneau 10 bits ne sont pas étirés par le look. A49-04 et A21-03 restent vrais pour la netteté et le grain. |
| A49-05 | Toujours présent | Voir A41-01. |
| Renvois 09 et 49 vers 41 (statistiques sur l'image débandée) | Réfuté comme défaut | Sous RIFE, `analyze_source` lit l'image débandée (`upscale_stage.cpp:293-297`) ; sous DistilDRBA, l'image non débandée (`:289-291`, puis `:297`) : les statistiques sont bien prises dans les deux cas. Le débandage déplace chaque canal de 3/1000 du code au plus, à moyenne nulle (`deband_kernels.h:13-17`) : sans effet mesurable sur la luminance moyenne, les centiles ou les bandes noires. |
| Renvoi 30 vers 41 (thread des LUT sans `try`) | Confirmé | A41-09. |
| Renvoi 42 vers 41 (métadonnées fixes, seul le grade plafonné) | Réfuté | Les métadonnées suivent le plafond : `match_kwin` lit `grade->params().peak` (`placebo_grade.cpp:217`), qui porte déjà les 250 nits de `chain_hdr.cpp:252`, et réannonce dès que le maximum change (`:233-243`). C'est GPU-07, toujours présent. Elles ne sont fixes (et absentes) qu'avec `--no-hdr-metadata`. |
| Renvoi 42 vers 41 (statistiques sur 9 216 points) | Réfuté comme défaut pour le grade | `frame_stats.h:5-6`, `:32-33` : une grille de 128 × 72 points lus sans moyenne. Le grade n'en tire que la luminance moyenne et la part claire (`SceneLight`). L'erreur d'échantillonnage vaut environ 1 % de l'écart-type de l'image pour la moyenne et 0,5 point au plus pour la part claire, sous les seuils de `light_tick` (3 % et 2 points, `chain_hdr.cpp:294-295`). Et la lumière de scène ne déplace les émetteurs que de 8 % à l'écran (A41-02). Un petit émetteur de moins de 15 pixels peut échapper à la grille : cela touche la détection des flashs, qui relève de l'audit 42. |
| A30-04 | Toujours présent | Même constat. |

## Renvois

| Vers l'audit | Où | Constat |
|---|---|---|
| 10 | `src/app/play/placebo_screen.cpp:155` | Commentaire « netteté sur MAIN » faux (A41-06). |
| 32 | `src/render/shaders/anime-hdr-anime/3-anime.hook:60-130`, `src/render/shaders/anime-hdr-look.hook:128-147` | Complète A32-05 : en HDR, trois passes plein écran à la résolution de sortie (anime, look, kwin), quatre avec le vrai noir. Par pixel : 23 lectures et une douzaine de `pow` vectoriels pour la passe anime, 10 lectures 2D et 2 lectures 3D pour le look. Plusieurs valeurs ne dépendent que des paramètres (`pq_of(0.55 * white)`, l'inverse de la matrice LMS) et sont recalculées à chaque pixel. À mesurer avec `grade_gpu_us`. |
| 38 | `docs/hdr-anime.md:63-67`, `src/render/grade_params.h:59-61`, `src/app/play/chain_hdr.cpp:381-383` | Trois promesses non tenues : détails des nuits lisibles (A41-03), blanc de la source à 203 nits (A41-02), vrai noir annoncé en SDR (A41-07). |
| 42 | `src/app/play/chain_hdr.cpp:321` | `next_flashing` décide du limiteur de flashs ; ses seuils n'ont pas été audités ici. |
| 43 | `src/app/play/placebo_screen.cpp:64-66` | `output_description` rend « sortie HDR10 (PQ BT.2020) » en français, repris par `output_text` (`chain_hdr.cpp:225-235`) : à vérifier s'il s'affiche. |
| 46 | `data/luts/ATTRIBUTION.md` | Auteur et licence des quatre LUT livrées non indiqués. |

## Points positifs

- `parse_cube` est strict : taille de fichier plafonnée avant lecture, taille de 17 à 65, valeurs finies, nombre de
  lignes exact, mots-clés inconnus refusés ; un corpus de fuzz existe.
- La courbe de sortie est monotone, et aucun canal n'atteint l'écrêtage de la passe kwin pour les couleurs SDR.
- Les passes du grade travaillent en FP16 (`find_fbo_format` de libplacebo) et le tramage vient après les hooks
  `OUTPUT` (`renderer.c:2833` puis `:2957`).
- `look_clean` est une vraie protection de qualité : le look donne le niveau du voisinage, le grain et les paliers
  gardent leur contraste.
- La seconde composition de la pile (65³ vers 65³) est exacte, et l'arrondi en demi-flottant reste à 0,25 code.
- Le pic automatique est borné de 200 à 2 000 nits (`host_facts.cpp:283-289`) : `decide_grade` ne refuse jamais
  un pic lu à l'écran.
- Le changement de grade attend l'image mesurée (`set_grade_after`), avec un délai de garde de 250 ms.

## Questions ouvertes

1. **Blanc à 70 % du pic** (A41-02) : est-ce le rendu voulu ? Hypothèse retenue ici : oui pour l'aspect, puisque
   la pile a été réglée à l'œil, mais le limiteur OLED et le curseur Éclats doivent alors agir après la pile.
2. **Ombres écrasées sous RVB 16** (A41-03) : voulu, ou effet de bord de la LUT de base ? Hypothèse retenue : à
   juger à l'œil sur des plans de nuit avant toute correction.
3. **Écran SDR** : le grade y garde l'image jusqu'à 90 % du blanc, puis comprime (un blanc de surface sort à
   95 %) pour laisser 5 % aux émetteurs. Ce gain vaut-il cinq passes et des blancs un peu gris ?
4. **Correction d'A41-01** : la chaîne dans le shader est la seule exacte. Son coût est à mesurer sur la 4070 Ti
   avant de choisir entre elle et une table de 129³.
