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
Ils apparaissent dans le menu (le menu relit le dossier à chaque ouverture de la liste).
- Format : LUT 3D, `LUT_3D_SIZE` de 17 à 65, `DOMAIN_MIN`/`DOMAIN_MAX` et `LUT_3D_INPUT_RANGE` lus ; les LUT 1D sont
  refusées ; un fichier illisible est refusé avec une ligne de journal « look : … ».
- **LUT SDR (par défaut)** : faite pour une image BT.709 en gamma (blanc = 1), comme la plupart des LUT de Resolve.
  Le lecteur ramène l'image à ce domaine (blanc de référence 203 nits, gamma 2,4), applique la LUT, puis remonte ;
  les éclats HDR au-dessus du blanc gardent leur niveau et reçoivent la teinte du look.
- **LUT d'HDR** : si le nom du fichier contient `PQ`, `HDR` ou `2020`, elle est lue comme étalonnée en PQ BT.2020
  (les codes PQ de l'image entrent directement dans la LUT). Utile pour les LUT « anime HDR » publiées sur
  YouTube. Sur une sortie SDR, une LUT d'HDR est lue comme une LUT SDR.
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
