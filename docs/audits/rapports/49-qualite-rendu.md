# Rapport de l'audit 49. Qualité du rendu

Branche : `claude/project-thread-1zjuya`. Commit audité : `8555f33`. Date : `2026-11-09`.

## Verdict

Les garde-fous de précision des réseaux sont sérieux (FP8 de l'interpolation refusé sur mesure, repli FP16 à chaque
refus), mais la qualité est cédée ailleurs, en silence et sans mesure : l'Auto ne joue jamais le meilleur SPAN,
réduit une source 1080p avant de l'agrandir, écrit l'anneau en 10 bits sans la porte prévue, et la pile de looks du
grade est approchée par une table dont l'écart, mesuré ici, atteint 15 codes PQ 10 bits.

## Cadrage retenu

Demande de Tatoub (09/11) : vérifier pour de bon que la qualité du rendu n'est négligée nulle part ; perdre de la
qualité pour gagner de la vitesse ne doit presque jamais se faire. Aucune question de cadrage posée, sur consigne :
tout est audité au maximum.

- Cible : RTX sous TensorRT, Windows 11 et CachyOS ; la RTX 4070 Ti de Tatoub (rang « Rapide »), écran 1440p 165 Hz.
- RIFE 4.25, RIFE 4.22 lite et DistilDRBA comptent autant. Le VRR est hors sujet.
- Règle appliquée : une baisse de qualité faite pour la vitesse est un défaut, sauf mesure qui la justifie. Pour
  chacune : le gain est-il mesuré, et la correction tient-elle la cadence ?
- Les défauts déjà relevés par les audits 02, 03, 07, 09 et 15 ne sont pas recopiés : ils sont revérifiés plus bas.
- Le fil n'a ni GPU NVIDIA ni moteurs construits. Une seule mesure a pu être faite ici (A49-05, code CPU du dépôt).
  Tout ce qui demande la carte est marqué « à confirmer », avec la manipulation.

Le chemin suivi : décodage, conversion hôte, ingestion YUV vers RVB FP16, réduction à la hauteur de travail,
deband, mouvement (RIFE ou DistilDRBA), SPAN ×2, warp, anneau, libplacebo (netteté, grade, tramage), écran.

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A49-01 | Majeur | `src/app/play/auto_config.cpp:69-75`, `src/app/play/options.h:35-36`, `src/app/play/model_choice.h:33-37` | L'Auto ne joue jamais le meilleur SPAN. `set_span` vide `out.model` à tous les rangs, et le modèle par défaut est le plus léger du catalogue (`…Performance_SPANF3_b5f48…`). Le b8f64 (« Balanced »), meilleur, n'est pris que par l'Auto calibré, et alors en FP8 (A49-03). Même le rang Ultra (RTX 4080 à 5090) joue le b5f48. Aucune mesure du dépôt ne dit ce que le b5f48 perd face au b8f64, ni que le b8f64 FP16 ne tient pas. | RTX 4070 Ti ou RTX 4090, Auto sans calibration : SPAN b5f48. Or avec le warp après SPAN il n'y a qu'un SPAN par image source (24 par seconde, 41,7 ms de budget), et le b8f64 FP16 est mesuré à 14,25 ms en 1088×1920 (`docs/span-fp8-cutlass.md:14-37`), soit environ 6,3 ms en 720p. | Poser le b8f64 FP16 dès le rang où il tient (Rapide et Ultra au moins), et ne descendre au b5f48 que sur mesure. | Lu. À confirmer : sur la 4070 Ti, binaire `linux-release`, lecture 1080p 24 i/s à 165 Hz avec `--model=…Balanced…b8f64…fp16.onnx`, relever images perdues et facteur tenu ; puis PSNR et SSIM de b5f48 contre b8f64 sur `calibration.mkv`. |
| A49-02 | Majeur | `src/app/play/auto_config.cpp:48`, `:137` ; `src/app/play/machine_calibration.cpp:252` ; `src/app/play/settings.h` (`work_height = 720`) | La source est réduite avant SPAN, puis réagrandie. La hauteur de travail vise la moitié de l'écran (`ideal = display_height / 2`), plafonnée par `kWorkCap{720,540,720,720,1080}` ; la calibration n'essaie rien au-dessus de la moitié de l'écran. Une source 1080p perd donc plus de la moitié de ses pixels avant le réseau, qui doit les réinventer. Aucun document ne justifie ce choix par une mesure de qualité. | (1) Écran 1440p, 4070 Ti : 1080p réduit à 720p (Hermite), SPAN rend 1440p. (2) Écran 4K, 4070 Ti : 720p, SPAN rend 1440p, puis Lanczos ×1,5 jusqu'à 2160p, alors que SPAN 1080p vers 2160p tient à 24 i/s d'après `docs/span-fp8-cutlass.md:205-207` (15,5 ms). (3) Écran 1080p, rang Normal ou Auto calibré : 1080p réduit à 540p puis remonté à 1080p. | Sur un écran de 1440 lignes ou plus, comparer sur la carte « 1080p, SPAN, réduction à l'écran » à « 720p, SPAN » et garder le meilleur qui tient. Sur écran 4K, travailler en 1080p dès que la mesure le permet. Sur écran 1080p avec source 1080p, voir le compromis C2. | Lu. À confirmer : PSNR et SSIM contre un rendu de référence (source 1080p, b8f64 FP32, réduction Spline36 à l'écran), pour chaque hauteur de travail, sur `calibration.mkv`. |
| A49-03 | Majeur | `src/app/play/auto_config.cpp:167-169`, `:202` ; `src/app/play/machine_calibration.h:37-47` | L'Auto calibré pose SPAN en FP8 alors que la décision est suspendue. `docs/adr/0007-span-basse-precision-par-defaut.md` (statut) et le commentaire des lignes 167-169 disent que les préréglages ne posent plus le FP8 tant que sa porte n'est pas mesurée ; la ligne 202 pose pourtant `Performance` dès que la calibration a retenu un modèle. La calibration n'essaie que « b8f64 FP8 » puis « b5f48 FP16 » : jamais le b8f64 FP16. Le gain réel du FP8 en lecture est de 6,5 % (médiane 15,53 contre 14,5 ms à 24 i/s, `docs/span-fp8-cutlass.md`), pas les 37 % du banc : le gain est en watts. | Linux, RTX 40, calibration faite : b8f64 FP8 (49 à 52 dB contre l'original). Windows, où le FP8 est éteint : b5f48 FP16. Dans les deux cas le b8f64 FP16, plus fidèle et à peine plus lent, n'a pas été essayé. | Essayer le b8f64 FP16 en premier ; ne garder le FP8 que s'il fait tenir une chaîne que le FP16 ne tient pas. Accorder le commentaire et l'ADR 0007. | Lu. À confirmer : rejouer la calibration avec les trois candidats et comparer ce qui tient. |
| A49-04 | Majeur | `src/app/play/options.h:113-114`, `src/app/play/player_run.cpp:171-175`, `src/gpu/cuda/backend/ring_slot.h:34` | L'anneau est en RGB10A2 par défaut sous TensorRT sans la porte de banding prévue. `docs/adr/0008-anneau-rgb10a2.md` (point 5) le reconnaît : « le défaut a changé sans la porte de l'étape 5.5c ». Le commentaire de `player_run.cpp:171` dit encore l'inverse. La sortie FP16 de SPAN est donc coupée à 10 bits (tramage TPDF de ±1 code) **avant** la netteté, la pile de looks et l'expansion HDR, qui étirent les paliers et amplifient le bruit de tramage. Le gain annoncé est de la mémoire (moitié moins) et « −25 % » sur les noyaux d'écriture, jamais mesuré. | Plan sombre en dégradé, grade HDR anime actif : l'expansion des hautes lumières et les LUT de contraste multiplient la pente locale, un palier de 10 bits devient visible, ou le grain du tramage monte. | Revenir à RGBA16 par défaut tant que `tools/ring/banding.py` n'a pas passé la porte ; RGB10A2 seulement quand la VRAM manque. | Lu. À confirmer : `tools/ring/banding.py` sur les plans sombres de `calibration.mkv`, avec et sans `--no-ring-rgb10a2`, grade actif (seuil de l'ADR : ≤ 1,10 × la référence). |
| A49-05 | Majeur | `src/render/cube_lut.cpp:289-314`, `src/render/cube_lut.h:19`, `:38-42` ; `src/app/play/look_catalog.cpp:74-123` | La pile de looks par défaut (11 lectures de LUT : base HDR, contrastes, hautes lumières, peau) est composée en une seule table de 65³ lue en trilinéaire. Le commentaire promet « sans écart », ce qui n'est vrai que pour une pile d'une seule LUT. Mesuré ici sur 400 000 couleurs SDR portées en PQ BT.2020 (203 nits), contre la chaîne exacte des 11 lectures : écart maximal 15,5 codes PQ 10 bits (6,1 près des gris), moyenne 0,89, 28,6 % des couleurs à plus d'un code, 11,6 % à plus de deux. À 129 points : maximum 7,5, moyenne 0,26, 4,6 % à plus d'un code. Le gain de vitesse de la composition n'est mesuré nulle part. | Toute lecture avec le grade HDR anime : les teintes saturées et les transitions de la courbe de contraste sont décalées de plusieurs codes, de façon non uniforme (risque de paliers dans les dégradés colorés). | Composer à 129 points (environ 17 Mo en rgba16hf), ou composer en tétraédrique, ou enchaîner les LUT dans le shader si la mesure montre que le coût est négligeable. Corriger le commentaire. | Mesuré (programme hors dépôt qui appelle `compose_luts` et `sample_lut`). À confirmer sur la carte : ΔE_ITP au p99,9 sur `tools/grade/shots30.tsv`, et coût de la passe en 65, 129 et chaîne. |
| A49-06 | Majeur | `src/gpu/cuda/backend/warp_kernels.cu:144-171`, `:202-211` ; `src/app/play/settings.h:106-117` ; `docs/tests-pc.md:193-203` | Avec le warp après SPAN (toujours le cas en Auto, A09-08), chaque image créée est un rééchantillonnage **bilinéaire** de la sortie de SPAN, par le filtrage matériel (fraction sur 8 bits, tolérance doublée à 4/1024). À 165 Hz, six images affichées sur sept passent par là : elles sont plus douces que les images sources, et SPAN ne nettoie plus les défauts du warp (`docs/adr/0002-span-une-fois-par-source.md` le dit). Le gain du warp filtré reste « À mesurer » dans `docs/tests-pc.md`. | Panoramique lent sur des traits fins à ×6,875 : la netteté pulse entre l'image source (nette) et les images créées (adoucies par le bilinéaire à position fractionnaire). | Mesurer un warp bicubique (Catmull-Rom, 16 lectures ou 4 lectures bilinéaires) : coût attendu bien sous la milliseconde par image. Garder le bilinéaire seulement si le bicubique fait perdre la cadence. | Lu. À confirmer : PSNR des images créées contre le chemin « warp avant SPAN » et contre un warp bicubique, sur trois plans à mouvement lent ; `ncu` ou chronos du lecteur pour le coût. |
| A49-07 | Mineur | `src/app/play/span_int8.cpp:194-202`, `src/app/play/span_int8.h:39-45` | Au cran Équilibre, annoncé « FP16 », `span_variant_order` prend une variante INT8 dès qu'un reçu existe (`tier != Quality && caps.int8`). Le libellé ment sur la précision jouée. | Un reçu INT8 laissé par un essai du cran Performance ou par un outil : le cran Équilibre joue ensuite l'INT8 (porte 42/38 dB), sans le dire. | Ne prendre l'INT8 qu'au cran Performance. | Lu. À confirmer : qui écrit ce reçu en usage normal (si personne, le défaut est dormant). |
| A49-08 | Mineur | `src/gpu/span_fp8/derive.h:17-37`, `src/gpu/cuda/backend/tensorrt_span_derive.cpp:14-16`, `:305` | La porte du FP8 a été réglée pour laisser passer le modèle. Les seuils du p99,9 valent 1,25 × ce que le b8f64 a mesuré (10/255 et 4,5/255 dans le sombre), parce que les seuils d'origine (4/255 et 2/255) « refusaient tout ». La référence est le FP16 des mêmes noyaux, pas le modèle FP32 d'origine : une erreur commune aux deux chemins est invisible. Les neuf cas durs ne décident pas. | Un pixel sur mille peut s'écarter de 10 codes 8 bits (4,5 dans le sombre, là où le grade étire) et la porte accepte. | Juger contre l'ONNX FP32 d'origine, après le grade, en ΔE_ITP (ce que propose déjà l'ADR 0007) ; faire décider les cas durs. | Lu. |
| A49-09 | Mineur | `src/app/play/upscale_motion.cpp:97-193`, `src/kernels/spec/rife.h` (`kSingleRunMotion720 = 15`), `src/app/play/auto_config.cpp:166` | Avec les poses (défaut de tous les préréglages), RIFE ne passe qu'une fois par paire, à t = ½ ; les autres instants sont extrapolés par le warp tant que le mouvement de la paire précédente reste sous 15 px. Le commentaire dit « mesuré A/B : qualité équivalente », sans chiffre ni manipulation dans le dépôt. Un mouvement non linéaire sous 15 px (rotation, zoom, accélération) est rendu comme linéaire. | Rotation lente d'un personnage à ×6,875 : six instants tirés d'un seul flot au milieu. | Écrire la mesure (PSNR des instants extrapolés contre RIFE à l'instant exact) et régler le seuil sur elle. | Lu. À confirmer : la même vidéo avec et sans poses, export des images créées, PSNR entre les deux et contre l'image vraie d'une source 60 i/s décimée. |
| A49-10 | Mineur | `src/app/play/chain_setup.cpp:27`, `:359-363` | RIFE estime le mouvement à 720p dès que le travail dépasse 720p (`kRifeMotionHeight`), et le flot est remonté. C'est le pendant, pour RIFE, d'A03-19 (DistilDRBA). `docs/adr/0002` le décide sans mesure de qualité. Sans effet tant que le travail reste à 720p ; il le deviendra si A49-02 est corrigé. | Travail en 1080p : flot estimé à 720p, bords des objets fins moins précis. | Mesurer RIFE à 1080p contre 720p (coût et PSNR) avant de fixer la règle. | Lu. À confirmer : même manipulation que A03-19. |
| A49-11 | Mineur | `src/app/play/upscale_stage.cpp:498-504`, `:735-738` ; `src/kernels/spec/detect.h` (`kHeldThreshold = 70`) | Un dessin jugé « tenu » (écart sous 0,70 niveau par bloc de 16×16) réutilise la sortie SPAN précédente, jusqu'à deux fois de suite. Un petit changement réel (clignement lointain, particule de 2×2 pixels, grain animé) est gelé. Le gain n'est chiffré nulle part. | Plan fixe avec neige fine ou scintillement : les détails sautent une image sur deux ou trois. | Mesurer le gain ; à défaut, juger sur le maximum du bloc plutôt que sur sa moyenne. | Lu. À confirmer : compter les images réutilisées sur un épisode et vérifier à l'œil trois plans à particules. |
| A49-12 | Mineur | `src/app/play/placebo_screen.cpp:130`, `:148-154`, `:320` | La présentation part de `pl_render_default_params` : mise à l'échelle Lanczos sans anti-ringing vers le haut, Hermite vers le bas ; l'EWA Lanczos n'existe que par choix manuel (`final_scaler`), et le deband est éteint par défaut. Sans effet quand l'anneau a la taille de l'écran (1440p chez Tatoub) ; touche les écrans 4K (A49-02) et les fenêtres. | Écran 4K : SPAN rend 1440p, Lanczos ×1,5 sans anti-ringing, halos sur les traits noirs. | Poser l'anti-ringing et mesurer le coût d'`ewa_lanczossharp` (une passe à la taille de l'écran). | Lu ; les valeurs par défaut de libplacebo v7.360.1 sont citées de mémoire. À confirmer : les relire dans `renderer.c`, puis chronométrer la passe. |
| A49-13 | Mineur | `src/media/frame_data.cpp:247-275` | La conversion hôte des sources 4:2:2, 4:4:4 et 12 bits utilise `SWS_BILINEAR` sans `SWS_ACCURATE_RND` ni `SWS_FULL_CHR_H_INT`, et ramène le 12 bits à 10. Complète A15-07 (la chroma 4:4:4 jetée) : même en gardant le 4:2:0, le filtre est le moins bon de swscale. | Blu-ray 4:4:4 ou source 12 bits : chroma adoucie et arrondie avant SPAN. | Garder le 4:4:4 jusqu'à l'ingestion (A15-07) ; à défaut, Lanczos ou Spline et arrondi exact, 16 bits en sortie. | Lu. |
| A49-14 | Mineur | `tools/rife/convert_rife_fp16.py`, `docs/fp8-interpolation.md` | Aucune mesure écrite du RIFE FP16 et du DistilDRBA FP16 livrés contre leurs originaux FP32 (le FP8 a la sienne). Les îlots FP32 du script sont un bon choix, mais rien ne prouve qu'ils suffisent, et A03-18 montre qu'un GridSample de DRBA y a échappé. | Une régression de conversion passerait sans alerte (A03-02 : les tests de concordance fusionnés sont cachés). | Ajouter la mesure au script de conversion et l'écrire dans `docs/interpolation.md`. | Lu. À confirmer : PSNR du flot et de l'image créée, FP16 contre FP32, sur dix paires. |

## Compromis qui restent, à décider par Tatoub

Chaque ligne est une baisse de qualité acceptée aujourd'hui. « Mon avis » dit ce que je ferais ; la décision est à
toi. « Tient ? » dit si la correction garde la cadence d'après ce qui est mesuré.

| N° | Compromis | Ce qu'on perd | Ce qu'on gagne | Tient ? | Mon avis |
|---|---|---|---|---|---|
| C1 | SPAN b5f48 au lieu du b8f64 en Auto (A49-01) | Le meilleur modèle n'est jamais joué | Environ 6 ms par image source en 720p (estimé) | Très probable sur 4070 Ti et plus ; à mesurer | Corriger : b8f64 FP16 aux rangs Rapide et Ultra |
| C2 | Source 1080p réduite à 720p ou 540p avant SPAN (A49-02) | Les détails réels de la source | Un SPAN 2,25 à 4 fois moins cher | 1080p tient à 24 i/s sur 4070 Ti d'après le banc ; à vérifier avec l'interpolation | Corriger sur écran 4K ; mesurer sur 1440p ; sur écran 1080p, à toi de dire si « Restaurer les traits » vaut la perte |
| C3 | SPAN FP8 en Auto calibré (A49-03) | 49 à 52 dB au lieu du FP16 | 6,5 % en lecture, moitié moins de watts | Oui, le FP16 est à 1 ms près | Corriger : FP16 d'abord, FP8 seulement s'il fait tenir un cran de plus |
| C4 | RIFE 4.22 lite imposé par l'Auto calibré (renvoi 03 vers 07) | RIFE 4.25 sur les cartes qui le tiennent | Rien de mesuré | À mesurer | Corriger : calibrer avec le modèle du rang |
| C5 | Anneau en 10 bits tramés (A49-04) | 6 bits avant le grade | Moitié de la VRAM de l'anneau | Oui (RGBA16 était le défaut) | Revenir à RGBA16 jusqu'à la porte |
| C6 | Pile de looks approchée en 65³ (A49-05) | Jusqu'à 15 codes PQ, 0,9 en moyenne | Une lecture de texture au lieu de onze | Oui en 129 points (même coût par image) | Corriger en 129 points |
| C7 | Warp bilinéaire après SPAN (A49-06) | Netteté des images créées ; SPAN ne nettoie plus le warp | Deux SPAN de moins par image source à ×3 (ADR 0002) | Le bicubique : très probable. Le warp avant SPAN : non à 165 Hz | Garder « après SPAN », passer le warp en bicubique après mesure |
| C8 | Une seule passe RIFE par paire sous 15 px (A49-09) | Mouvements non linéaires lents | Jusqu'à cinq passes RIFE par paire à 165 Hz | Non sans cela à ×6,875 | Garder, mais écrire la mesure et régler le seuil |
| C9 | Mouvement estimé à 720p (A49-10, A03-19) | Précision du flot au-dessus de 720p | Environ la moitié du coût du mouvement | À mesurer | Garder en attendant la mesure |
| C10 | Dessins tenus : sortie SPAN réutilisée (A49-11) | Petits changements gelés | Un SPAN sur deux ou trois sur les plans fixes | Oui sans cela (le budget est prévu pour un SPAN par image) | Retirer, ou durcir le seuil |
| C11 | Chroma bilinéaire à l'ingestion TensorRT (A03-01) | Le contrat dit Spline36 | Rien de mesuré | Oui (un noyau par image source) | Corriger |
| C12 | Chroma 4:4:4 et 12 bits réduits à l'entrée (A15-07, A49-13) | La qualité des meilleures sources | Un seul format d'ingestion | Oui | Corriger après les autres (sources rares) |
| C13 | DistilDRBA : pas de deband avant SPAN, estimation à 720p, grille FP16 (A09-07, A03-19, A03-18) | Qualité du niveau « Très rapide » | Simplicité du chemin | À mesurer | Corriger A03-18 d'abord (pur défaut) |
| C14 | Porte FP8 réglée sur le modèle (A49-08) | Garantie réelle sur les pixels extrêmes | Le FP8 passe | Sans objet | Rejuger en ΔE_ITP après le grade |
| C15 | Réduction avant SPAN faite en lumière gamma, filtre Hermite | Léger assombrissement des traits fins | Rien (choix de simplicité) | Oui | À mesurer ; sans objet si C2 retire la réduction |
| C16 | Pas de niveau d'interpolation au-dessus de RIFE 4.25 (GMFSS retiré, ADR 0012) | Le meilleur rendu possible hors cadence | Un lecteur simple | Non en direct | Garder : décision déjà prise |

## Défauts connus revérifiés

| ID | Statut | Note |
|---|---|---|
| A03-01 | Toujours présent | `src/gpu/cuda/backend/slice_kernels.cu:24-45` : chroma bilinéaire, `IngestSpec::chroma` ignoré. Première perte de la chaîne, sur toutes les sources 4:2:0. |
| A03-18 | Toujours présent | Grille FP16 des GridSample de DistilDRBA ; rien à ajouter. |
| A03-19 | Toujours présent | DistilDRBA estime à 720p ; A49-10 est le même choix pour RIFE. |
| A03-02 | Toujours présent | Tests de concordance fusionnés cachés, seuil de 30 dB pour RIFE : aucune porte de qualité de l'interpolation ne tourne en CI. |
| A02-01 | Toujours présent | La porte FP8 ne voit pas un NaN ; s'ajoute à A49-08. |
| TRT-15 | Toujours présent | Échelles FP8 calibrées sur la première vidéo ouverte ; l'ADR 0007 demande un corpus fixe. |
| A15-12 | Toujours présent | Corpus de calibration FP8 en chroma au plus proche. |
| A15-07 | Toujours présent | Chroma 4:4:4 réduite avant SPAN ; A49-13 ajoute le filtre et les 12 bits. |
| A09-02 | Toujours présent | Le cran Performance de l'interpolation coupe le moteur fusionné : plus lent sans gain de qualité. |
| A09-07 | Toujours présent | Pas de deband avant SPAN avec DistilDRBA (`upscale_stage.h:318-320`). |
| A09-08 | Toujours présent | Auto vaut toujours « Après SPAN » avec le travail à 720p ; A49-06 en donne le coût en qualité. |
| A07-01 | Toujours présent | La calibration ne joue pas la chaîne appliquée ; s'ajoute à A49-03. |
| Renvoi 03 vers 07 | Toujours présent | `auto_config.cpp:204` : RIFE 4.22 lite imposé par l'Auto calibré, même au rang qui donne RIFE 4.25 (compromis C4). |
| Renvoi 02 vers 34 | Toujours présent | FP8 éteint sous Windows : l'Auto calibré y joue le b5f48 FP16 (A49-03). |

## Renvois

| Vers l'audit | Où | Constat |
|---|---|---|
| 07 | `src/app/play/machine_calibration.h:37-47`, `.cpp:252` | Candidats de la calibration : ajouter le b8f64 FP16, les hauteurs au-dessus de la moitié de l'écran et le RIFE du rang (A49-01 à A49-03). |
| 10 | `src/app/play/placebo_screen.cpp:130` | Paramètres par défaut de libplacebo, pas d'anti-ringing (A49-12). |
| 23 | `src/app/play/auto_config.cpp:167-169`, `src/app/play/player_run.cpp:171`, `src/render/cube_lut.h:38-42` | Trois commentaires contredits par le code (FP8 des préréglages, RGB10A2 « derrière une option », LUT « sans écart »). |
| 32 | `tests/unit/` | Aucun test ne mesure une qualité de bout en bout (image de référence contre sortie de la chaîne) ; les portes existantes comparent un chemin rapide à un chemin voisin, jamais à l'original. |
| 34 | `src/app/play/span_auto.cpp:22-25` | Windows et Linux ne jouent pas le même SPAN en Auto calibré. |
| 41 | `src/render/cube_lut.cpp:289-314`, `src/app/play/look_catalog.cpp:74-123` | Écart de la LUT composée (A49-05) ; statistiques du grade sur l'image débandée (renvoi 09 vers 41). |

## Points positifs

- Le FP8 de RIFE est refusé sur mesure (31 dB au pire, `docs/fp8-interpolation.md`) et celui de DistilDRBA reste
  éteint : la règle de Tatoub y est déjà appliquée.
- Tout refus d'une variante basse précision retombe sur le FP16 ; les reçus portent la carte, la version des noyaux
  et la taille.
- `trtexec` construit en `--builderOptimizationLevel=5` et en typage strict : la précision est celle de l'ONNX, sans
  surprise de TensorRT (`src/prepare/trtexec_command.cpp:67-83`).
- La chaîne reste en RVB FP16 non borné de l'ingestion à l'écriture de l'anneau ; aucun passage par 8 bits.
- Le deband se fait avant SPAN avec un motif partagé, et le deband de libplacebo devient sans grain sous le grade.
- Le tramage final de libplacebo vient après les hooks du grade.
- Le script de conversion de RIFE garde en FP32 la chaîne de la grille, GridSample et Resize.
- Sur l'écran de Tatoub (1440p, travail 720p), l'anneau a la taille de l'écran : aucune mise à l'échelle finale.

## Questions ouvertes

1. **« Restaurer les traits » sur un écran 1080p** : réduire une source 1080p à 540p pour que SPAN la redessine est
   un choix d'aspect, pas seulement de vitesse. Faut-il le garder par défaut, ou afficher la source 1080p telle
   quelle (éventuellement SPAN en 1080p puis réduction) ? Option la plus probable retenue ici : c'est un défaut de
   qualité à mesurer (A49-02, C2).
2. **Existe-t-il un SPAN AnimeJaNai plus lourd que le b8f64** (variante « Quality ») qui tiendrait à 24 SPAN par
   seconde sur une 4070 Ti ? Le catalogue (`model_choice.h:33-37`) n'en a pas ; à vérifier chez l'auteur du modèle.
3. **Seuils à confirmer à l'œil** : 15 px pour la passe unique de RIFE (A49-09), 0,70 niveau pour les dessins tenus
   (A49-11).
4. **Ordre des corrections proposé**, du plus sûr au plus coûteux : A49-04 (une ligne), A49-05 (une constante),
   A49-03, A49-01, A03-01, A49-06, puis A49-02 qui demande le plus de mesures.
