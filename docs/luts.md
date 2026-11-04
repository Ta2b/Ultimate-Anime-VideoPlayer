# Looks (LUT 3D .cube)

Le menu **Couleurs** propose un **Look (LUT)** et son **intensité** (0 à 100 %). Le look passe après le grade
(contraste, éclats, couleurs…) et avant le vrai noir OLED ; il agit sur toutes les routes (RTX HDR, HDR adaptatif,
SDR, sortie SDR). Intensité à 0 : la passe est sautée, l'image est identique octet pour octet.

## Looks du lecteur (« maison »)
Produits par le calcul (`src/render/looks.cpp`), sans fichier ni licence tierce : **Éclatant** (`maison:vif`),
**Ciel froid** (`maison:shinkai`, ombres froides, bleus et cyans saturés), **Pastel chaud** (`maison:pastel`, noirs
relevés, tons chauds), **Aplats nets** (`maison:aplats`, contraste fort, couleurs d'aplat franches). Chacun est une
LUT de 33 points.

## Empiler plusieurs LUT (comme les nœuds de Resolve)
Le menu Couleurs empile les looks : le bouton **+** ajoute un emplacement à la fin (jusqu'à **quinze** en tout), le bouton **−** retire le dernier. Les emplacements s'appliquent **dans l'ordre** (1, puis 2, puis 3...) ; chacun a sa
propre intensité de 0 à 100 % (0 : l'emplacement est sauté). Chaque LUT lit le résultat de la précédente. Si l'une
des LUT est une LUT d'HDR (nom avec PQ, HDR ou 2020), toute la chaîne travaille sur les codes PQ BT.2020, comme une
timeline HDR de Resolve ; sinon sur l'image SDR BT.709. Ordre habituel avec une LUT « de base » d'expansion SDR vers
HDR : le look créatif d'abord (emplacement 1), la LUT de base en dernier (le dernier emplacement), ou l'inverse selon ce que
la LUT de base attend en entrée ; les deux ordres se comparent en échangeant les emplacements.

## LUT d'HDR avec une zone noire (« HDR grading base »)
Certaines LUT d'HDR rendent du noir pur pour les couleurs hors de leur gamut d'entrée (un tiers du cube pour la
« HDR grading base ver.9.0 33 point »). Entre la zone utile et ce noir, l'interpolation assombrirait les pixels du
bord : au chargement, ces entrées sont recopiées du voisin valide le plus proche (le journal dit combien). Les
couleurs d'une image d'anime n'atteignent presque jamais cette zone ; la correction ne touche que ses bords.

## Tes propres LUT
Dépose des fichiers `.cube` dans le dossier `luts` des données du lecteur (Linux : `~/.local/share/ajp/ajp/luts`, à côté de `play.toml` ; le lecteur le crée à la première lecture).

Le lecteur livre aussi les 55 LUT d'HDR de `data/luts/` dans un dossier `luts/` à côté de lui (Contents/Resources dans AJP.app), lu après celui de l'utilisateur : un fichier du même nom dans le dossier de l'utilisateur prend le pas.
Ils apparaissent dans le menu (le menu relit le dossier à chaque ouverture de la liste).
- Format : LUT 3D, `LUT_3D_SIZE` de 17 à 65, `DOMAIN_MIN`/`DOMAIN_MAX` et `LUT_3D_INPUT_RANGE` lus ; les LUT 1D sont
  refusées ; un fichier illisible est refusé avec une ligne de journal « look : … ».
- **LUT SDR (par défaut)** : faite pour une image BT.709 en gamma (blanc = 1), comme la plupart des LUT de Resolve.
  Le lecteur ramène l'image à ce domaine (blanc de référence 203 nits, gamma 2,4), applique la LUT, puis remonte ;
  les éclats HDR au-dessus du blanc gardent leur niveau et reçoivent la teinte du look.
- **LUT d'HDR** : si le nom du fichier contient `PQ`, `HDR` ou `2020`, elle est lue comme étalonnée en PQ BT.2020
  (les codes PQ de l'image entrent directement dans la LUT). Utile pour les LUT « anime HDR » publiées sur
  YouTube. Sur une sortie SDR, une pile qui contient une LUT d'HDR est coupée en entier (elle serait lue comme du SDR et donnerait des couleurs fausses).
- Une LUT exportée de DaVinci Resolve (Fichier > Exporter la LUT, 33 ou 65 points) convient.
- Évite les LUT de conversion Log vers Rec.709 : l'anime est déjà en Rec.709, elles écraseraient l'image.

## Origine et licence
| Look | Origine | Licence |
|---|---|---|
| `maison:vif`, `maison:shinkai`, `maison:pastel`, `maison:aplats` | calculés par ce dépôt | ceux du dépôt |
| Fichiers de l'utilisateur | déposés par lui | les siens |

Aucune LUT de tiers n'est livrée avec le lecteur. Candidats trouvés pour la comparaison (non téléchargés ici, licences
à relire à la source avant toute redistribution) : packs Q-DDL (annoncés CC BY 4.0, attribution à Q-DDL), packs gratuits
de FreeVisuals (gratuit pour usage personnel et commercial selon leur page), « HDR grading LUT » de la chaîne YouTube
« 4K HDR anime channel » (licence non précisée : à télécharger soi-même pour essai, ne pas redistribuer).

## Idées de sdr2hdr (MIT) à réécrire, pas encore faites
Modes de ton « vivid » (blanc SDR au pic) et « reference » (BT.2408, 203 nits), styles « natural », « cinematic »,
« night » en préréglages du grade, choix de l'EOTF d'entrée (BT.1886 ou sRGB), dither à la quantification PQ.

Avec `--grade-ring` (essai), la pile de LUT composée est appliquée par le GPU du moteur, dans l'anneau, juste
après le grade : même pile (65 points, intensités comprises), même ordre, lue en flottant au lieu du demi-flottant
de la texture de l'écran. Rien ne change dans le menu ni dans `play.toml`.

Dans `play.toml`, la première LUT est `grade.look` / `grade.look_mix`, les suivantes `grade.looks` (noms) et `grade.looks_mix` (intensités), dans l'ordre.

## Grade adaptatif à la luminosité de la scène
Les curseurs **noirs profonds** et **couleurs** du grade et le **gris moyen de RTX HDR** valent pour une **scène sombre**.
Le bloc « Adaptatif à la luminosité de la scène » (réglages avancés, onglet Couleurs) donne les valeurs d'une **scène
claire** (noirs, couleurs, gris moyen) et deux seuils en pourcentage de luminance (13 % et 54 % par défaut, mesurés sur
l'image source). Le moteur TensorRT mesure sur le GPU la luminance moyenne de chaque image avant RTX HDR (un point sur
seize, une image de retard, sans copie) ; le lecteur la lisse sur environ 1,2 s, la remet à la mesure à chaque coupe de
scène, et glisse de la scène sombre à la scène claire : `t = smoothstep(bas, haut, luminance)`. Noirs et gris moyen
suivent t ; les couleurs ne s'effacent que sur les derniers 20 % de la transition. Seuils et valeurs par défaut tirés
des cinq scènes de Re:Zero ép. 18 de Tatoub (`~/ajp-reports/parametres-tatoub-2026-10-30/scenes/LISEZMOI.md` : Y vidéo
44 = sombre, 135 = très clair, soit 13 % et 54 % de luminance en RGB complet). Sans RTX HDR (HDR adaptatif, sortie SDR
gradée : tous les GPU), le shader du grade fait la même chose avec le code PQ moyen très lissé de l'image qu'il reçoit
(la source convertie ; seuils convertis en codes PQ, gris moyen = la courbe TrueHDR émulée). Pas encore adaptatifs : le contraste et l'intensité des LUT.

## Séparer une LUT d'HDR en courbe et couleur (chantier 8.3)
`tools/look/split_hdr_lut.py` lit une LUT PQ BT.2020 (ou une pile, `--compose`), en tire la courbe neutre (axe
des gris, 4 096 points, monotone) et la couleur (Ct/Cp de la LUT, intensité ramenée par l'inverse de la courbe),
réécrite dans le domaine SDR du lecteur avec `# ajp-domain: sdr`. `--check` vérifie que l'identité et une pure
courbe rendent une couleur identité (ΔE_ITP ≤ 0,1). Le résultat pour la pile par défaut est dans
`data/luts-v2/` (LISEZMOI.md).

Essai `--hdr-contract-v2` (chantier 8.4) : la pile par défaut, si l'utilisateur ne l'a pas changée, devient le seul
`ver9-couleur.cube` (SDR, sa courbe neutre laissée au grade), et la sortie HDR10 annonce les métadonnées du contrat
d'écran (C_pk, MaxFALL et noir en unités du contenu, primaires de l'EDID ou Display P3) au lieu de laisser KWin
compresser ; `--hdr-metadata=off` n'annonce rien. `play.toml` n'en garde rien.
Avec `--grade-ring` en plus (TensorRT), le grade de l'anneau devient une seule lecture 3D par pixel : la LUT de la
courbe du plan (look SDR compris), cuite en fond pour le plan courant et les deux suivants, datée du début de son
plan (la bascule se fait à l'image près), deux LUT mêlées dans un fondu. Sans plans connus (pré-analyse coupée),
le grade d'avant reste.
Une source HDR10 (8.9) annonce en plus, plan par plan, sa lumière mesurée par la pré-analyse (maxSCL et moyenne,
en métadonnées HDR10+) : la spline de libplacebo la ramène à C_pk sans détection de pic, un plan sous C_pk passe
tel quel et un plan plus clair est compressé.
L'apprentissage (8.10) suit la courbe par plan sous cet essai : les corrections des curseurs passent par une table
de migration (gris → décalage E par classe de scène sombre, éclat, claire ou ciel, ±0,25 ; noirs → contraste du
pied ; couleurs → vibrance, au plus 1,15) et s'écrivent dans `adaptive-learning-v2.tsv` ; les échantillons
d'`adaptive-learning.tsv`, faits pour l'autre grade, sont écartés et comptés au journal (app/play/grade_taste.h).
Les portes de 8.11 se vérifient hors du lecteur : `tools/grade/sim/simulate.py` passe une liste de plans (ou
`--synthetic 30`) par le portage Python de la courbe (`tools/grade/sim/curve.py`, comparé à la version C++ par
`--self-test`) et dit, plan par plan, E, T, le pic et l'APL affichés (pic ≤ C_pk + 0,5 %, APL ≤ 0,9 × MaxFALL) ;
`tools/grade/check_hdr.py` lit les instantanés PQ des 30 plans (compare_shots.sh) et vérifie pic, APL, ΔE_ITP sous K
contre la référence BT.2408 (moyenne ≤ 3, p95 ≤ 6), teinte (≤ 2°, ≤ 1° sur la peau), peau (≤ 0,5 diaphragme),
stabilité de l'APL d'un plan fixe (σ ≤ 0,5 %), images d'erreur aux coupes (0) et ColorVideoVDP (≥ −0,1 JOD, si
pycvvdp est installé).

## La pile par défaut en une seule LUT (04/11/2026)

La pile de LUT de Tatoub (la LUT de base `HDR grading base ver.9.0`, puis `HDR-contrast-Enhancement-L` ×7,
`Highlight-Enhancement-L` à 100 % et à 50 %, `skin fix` ; les emplacements à intensité 0 écartés) est cuite en une
seule LUT, `HDR pile unique 65 point.cube` (65³ points, PQ BT.2020), qui devient le look par défaut ; la liste des
LUT suivantes est vide par défaut (le menu Couleurs peut toujours en ajouter). Une pile choisie à la main dans un
`play.toml` existant n'est pas touchée.

    python3 -I tools/look/bake_stack.py --luts data/luts --out "data/luts/HDR pile unique 65 point.cube" \
      --play-toml play.toml --check

Le script refait la lecture du lecteur (interpolation trilinéaire, mélange `c + intensité × (lut(c) − c)`, entrées noires
d'un gamut sans usage remplacées par leurs voisins). **Ce n'est pas identique à la pile d'origine** : l'échantillonnage
à 65 points d'une pile très non linéaire donne, sur des images d'épisode, un écart moyen d'environ 1 code 10 bits (p99
de 3 à 5 codes, maximum d'environ 11 sur les couleurs les plus saturées). `--check` en donne les chiffres.
